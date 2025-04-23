import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from matplotlib.figure import Figure
from matrix.datasets.graph import KnowledgeGraph
from matrix.inject import _extract_elements_in_list, inject_object
from matrix.pipelines.matrix_generation.reporting_plots import ReportingPlotGenerator
from matrix.pipelines.matrix_generation.reporting_tables import ReportingTableGenerator
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output
from sklearn.impute._base import _BaseImputer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def enrich_embeddings(
    nodes: ps.DataFrame,
    drugs: ps.DataFrame,
    diseases: ps.DataFrame,
) -> ps.DataFrame:
    """Function to enrich drug and disease list with embeddings.

    Args:
        nodes: Dataframe with node embeddings
        drugs: List of drugs
        diseases: List of diseases
    """
    return (
        drugs.withColumn("is_drug", F.lit(True))
        .unionByName(diseases.withColumn("is_disease", F.lit(True)), allowMissingColumns=True)
        .join(nodes, on="id", how="inner")
        .select("is_drug", "is_disease", "id", "topological_embedding")
        .withColumn("is_drug", F.coalesce(F.col("is_drug"), F.lit(False)))
        .withColumn("is_disease", F.coalesce(F.col("is_disease"), F.lit(False)))
    )


def _add_flag_columns(
    matrix: pd.DataFrame, known_pairs: pd.DataFrame, clinical_trials: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Adds boolean columns flagging known positives and known negatives.

    Args:
        matrix: Drug-disease pairs dataset.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.

    Returns:
        Pairs dataset with flag columns.
    """

    def create_flag_column(pairs):
        pairs_set = set(zip(pairs["source"], pairs["target"]))
        # Ensure the function returns a Series
        result = matrix.apply(lambda row: (row["source"], row["target"]) in pairs_set, axis=1)

        return result.astype(bool)

    # Flag known positives and negatives
    test_pairs = known_pairs[known_pairs["split"].eq("TEST")]
    test_pair_is_pos = test_pairs["y"].eq(1)
    test_pos_pairs = test_pairs[test_pair_is_pos]
    test_neg_pairs = test_pairs[~test_pair_is_pos]
    matrix["is_known_positive"] = create_flag_column(test_pos_pairs)
    matrix["is_known_negative"] = create_flag_column(test_neg_pairs)

    # TODO: Need to make this dynamic
    # Flag clinical trials data
    clinical_trials = clinical_trials.rename(columns={"subject": "source", "object": "target"})
    matrix["trial_sig_better"] = create_flag_column(clinical_trials[clinical_trials["significantly_better"] == 1])
    matrix["trial_non_sig_better"] = create_flag_column(
        clinical_trials[clinical_trials["non_significantly_better"] == 1]
    )
    matrix["trial_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["non_significantly_worse"] == 1])
    matrix["trial_non_sig_worse"] = create_flag_column(clinical_trials[clinical_trials["significantly_worse"] == 1])

    return matrix


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            "target": Column(str, nullable=False),
            "is_known_positive": Column(bool, nullable=False),
            "is_known_negative": Column(bool, nullable=False),
            "trial_sig_better": Column(bool, nullable=False),
            "trial_non_sig_better": Column(bool, nullable=False),
            "trial_sig_worse": Column(bool, nullable=False),
            "trial_non_sig_worse": Column(bool, nullable=False),
        },
        unique=["source", "target"],
    )
)
@inject_object()
def generate_pairs(
    known_pairs: pd.DataFrame,
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    graph: KnowledgeGraph,
    clinical_trials: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    FUTURE: Consider rewriting operations in PySpark for speed

    Args:
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph: Object containing node embeddings.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.

    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    drugs_lst = drugs["id"].tolist()
    diseases_lst = diseases["id"].tolist()

    # Remove duplicates
    drugs_lst = list(set(drugs_lst))
    diseases_lst = list(set(diseases_lst))

    # Remove drugs and diseases without embeddings
    nodes_with_embeddings = set(graph._nodes["id"])
    drugs_lst = [drug for drug in drugs_lst if drug in nodes_with_embeddings]
    diseases_lst = [disease for disease in diseases_lst if disease in nodes_with_embeddings]

    # Generate all combinations
    matrix_slices = []
    for disease in tqdm(diseases_lst):
        matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
        matrix_slices.append(matrix_slice)

    # Concatenate all slices at once
    matrix = pd.concat(matrix_slices, ignore_index=True)

    # Remove training set
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1)
    matrix = matrix[~is_in_train]
    # Add flag columns for known positives and negatives
    matrix = _add_flag_columns(matrix, known_pairs, clinical_trials)

    return matrix


def make_batch_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
    batch_by: str = "target",
) -> pd.DataFrame:
    """Generate probability scores for drug-disease dataset.

    This function computes the scores in batches to avoid memory issues.

    FUTURE: Experiment with PySpark for predictions where model is broadcasted.
    https://dataking.hashnode.dev/making-predictions-on-a-pyspark-dataframe-with-a-scikit-learn-model-ckzzyrudn01lv25nv41i2ajjh

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """

    def process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        # Collect embedding vectors
        batch["source_embedding"] = batch.apply(lambda row: graph.get_embedding(row.source, default=pd.NA), axis=1)
        batch["target_embedding"] = batch.apply(lambda row: graph.get_embedding(row.target, default=pd.NA), axis=1)

        # Retrieve rows with null embeddings
        # NOTE: This only happens in a rare scenario where the node synonymizer
        # provided an identifier for a node that does _not_ exist in our KG.
        # https://github.com/everycure-org/matrix/issues/409
        removed = batch[batch["source_embedding"].isna() | batch["target_embedding"].isna()]
        if len(removed.index) > 0:
            logger.warning(f"Dropped {len(removed.index)} pairs during generation!")
            logger.warning(
                "Dropped: %s",
                ",".join([f"({r.source}, {r.target})" for _, r in removed.iterrows()]),
            )

        # Drop rows without source/target embeddings
        batch = batch.dropna(subset=["source_embedding", "target_embedding"])

        # Return empty dataframe if all rows are dropped
        if len(batch) == 0:
            return batch.drop(columns=["source_embedding", "target_embedding"])

        # Apply transformers to data
        transformed = apply_transformers(batch, transformers)

        # Extract features
        batch_features = _extract_elements_in_list(transformed.columns, features, True)

        # Generate model probability scores
        preds = model.predict_proba(transformed[batch_features].values)
        batch[not_treat_score_col_name] = preds[:, 0]
        batch[treat_score_col_name] = preds[:, 1]
        batch[unknown_score_col_name] = preds[:, 2]
        batch = batch.drop(columns=["source_embedding", "target_embedding"])
        return batch

    grouped = data.groupby(batch_by)

    result_parts = []
    for _, batch in tqdm(grouped):
        result_parts.append(process_batch(batch))

    results = pd.concat(result_parts, axis=0)

    return results


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
    batch_by: str,
) -> pd.DataFrame:
    """Generate and sort probability scores for a drug-disease dataset.

    FUTURE: Perform parallelised computation instead of batching with a for loop.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        treat_score_col_name: Probability score column name.
        not_treat_score_col_name: Probability score column name for not treat.
        unknown_score_col_name: Probability score column name for unknown.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset sorted by an additional column containing the probability scores.
    """
    # Generate scores
    data = make_batch_predictions(
        graph,
        data,
        transformers,
        model,
        features,
        treat_score_col_name,
        not_treat_score_col_name,
        unknown_score_col_name,
        batch_by=batch_by,
    )

    # Sort by the probability score
    sorted_data = data.sort_values(by=treat_score_col_name, ascending=False)

    # Add rank and quantile rank columns
    sorted_data["rank"] = range(1, len(sorted_data) + 1)
    sorted_data["quantile_rank"] = sorted_data["rank"] / len(sorted_data)

    return sorted_data


@inject_object()
def generate_reports(
    sorted_matrix_df: pd.DataFrame | ps.DataFrame,
    strategies: list[ReportingPlotGenerator | ReportingTableGenerator],
    file_suffix: str,
    **kwargs,
) -> dict[str, Figure | ps.DataFrame]:
    """Generate reporting plots.

    Args:
        sorted_matrix_df: DataFrame containing the sorted matrix
        strategies: List of reporting plot or table strategies
        file_suffix: Suffix for the file names
        kwargs: Extra arguments such as the drug and disease lists for tables

    Returns:
        Dictionary of plots or tables with strategy name as key
    """
    # Check that the names of the reporting strategies are unique
    names = [strategy.name for strategy in strategies]
    if len(names) != len(set(names)):
        raise ValueError("Reporting strategy names must be unique.")

    reports_dict = {}
    for strategy in strategies:
        reports_dict[strategy.name + file_suffix] = strategy.generate(sorted_matrix_df, **kwargs)

    return reports_dict
