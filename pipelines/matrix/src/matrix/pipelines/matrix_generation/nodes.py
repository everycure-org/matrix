"""Module with nodes for evaluation."""
import logging
from tqdm import tqdm
from typing import List, Dict, Union

from sklearn.impute._base import _BaseImputer

import pandas as pd

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix.datasets.graph import KnowledgeGraph

from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.pipelines.modelling.model import ModelWrapper


logger = logging.getLogger(__name__)


def enrich_embeddings(
    nodes: DataFrame,
    drugs: DataFrame,
    diseases: DataFrame,
) -> DataFrame:
    """Function to enrich drug and disease list with embeddings.

    Args:
        nodes: Dataframe with node embeddings
        drugs: List of drugs
        diseases: List of diseases
    """
    return (
        drugs.withColumn("is_drug", F.lit(True))
        .unionByName(
            diseases.withColumn("is_disease", F.lit(True)), allowMissingColumns=True
        )
        .withColumnRenamed("curie", "id")
        .join(nodes, on="id", how="inner")
        .select("is_drug", "is_disease", "id", "topological_embedding")
        .withColumn("is_drug", F.coalesce(F.col("is_drug"), F.lit(False)))
        .withColumn("is_disease", F.coalesce(F.col("is_disease"), F.lit(False)))
    )


@has_schema(
    schema={
        "source": "object",
        "target": "object",
    },
    allow_subset=True,
)
@inject_object()
def generate_pairs(
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    known_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    Args:
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        known_pairs: Labelled ground truth drug-disease pairs dataset.

    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    drugs_lst = drugs["curie"].tolist()
    diseases_lst = diseases["curie"].tolist()

    # Remove duplicates
    drugs_lst = list(set(drugs_lst))
    diseases_lst = list(set(diseases_lst))

    # Generate all combinations
    matrix_slices = []
    for disease in tqdm(diseases_lst):
        matrix_slice = pd.DataFrame({"source": drugs_lst, "target": disease})
        matrix_slices.append(matrix_slice)

    # Concatenate all slices at once
    matrix = pd.concat(matrix_slices, ignore_index=True)

    # Remove training set and return
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(
        lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1
    )
    return matrix[~is_in_train]


def make_batch_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
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
        batch["source_embedding"] = batch.apply(
            lambda row: graph.get_embedding(row.source, default=pd.NA), axis=1
        )
        batch["target_embedding"] = batch.apply(
            lambda row: graph.get_embedding(row.target, default=pd.NA), axis=1
        )

        # Retrieve rows with null embeddings
        # NOTE: This only happens in a rare scenario where the node synonymizer
        # provided an identifier for a node that does _not_ exist in our KG.
        # https://github.com/everycure-org/matrix/issues/409
        removed = batch[
            batch["source_embedding"].isna() | batch["target_embedding"].isna()
        ]
        if len(removed.index) > 0:
            logger.warning(f"Dropped {len(removed.index)} pairs during generation!")
            logger.warning(
                "Dropped: %s",
                ",".join([f"({r.source}, {r.target})" for _, r in removed.iterrows()]),
            )

        # drop rows without source/target embeddings
        batch = batch.dropna(subset=["source_embedding", "target_embedding"])

        # Apply transformers to data
        transformed = apply_transformers(batch, transformers)

        # Extract features
        batch_features = _extract_elements_in_list(
            transformed.columns, features, raise_exc=True
        )

        # Generate model probability scores
        batch[score_col_name] = model.predict_proba(transformed[batch_features].values)[
            :, 1
        ]

        return batch[[score_col_name]]

    # Group data by the specified prefix
    grouped = data.groupby(batch_by)

    # Process data in batches
    result_parts = []
    for _, batch in tqdm(grouped):
        result_parts.append(process_batch(batch))

    # Combine results
    results = pd.concat(result_parts, axis=0)

    # Add scores to the original dataframe
    data[score_col_name] = results[score_col_name]

    return data


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
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
        score_col_name: Probability score column name.
        batch_by: Column to use for batching (e.g., "target" or "source").

    Returns:
        Pairs dataset sorted by an additional column containing the probability scores.
    """
    # Generate scores
    data = make_batch_predictions(
        graph, data, transformers, model, features, score_col_name, batch_by=batch_by
    )

    # Sort by the probability score
    sorted_data = data.sort_values(by=score_col_name, ascending=False)
    return sorted_data


def add_flag_columns(
    matrix: pd.DataFrame, known_pairs: pd.DataFrame, raw_clinical_trials: pd.DataFrame
) -> pd.DataFrame:
    """_summary_.

    Args:
        matrix: _description_
        known_pairs: _description_
        raw_clinical_trials: _description_

    Returns:
        _description_
    """

    def create_flag_column(pairs):
        pairs_set = set(zip(pairs["source"], pairs["target"]))
        return matrix.apply(
            lambda row: (row["source"], row["target"]) in pairs_set, axis=1
        )

    # Flag known positives and negatives
    test_pairs = known_pairs[known_pairs["split"].eq("TEST")]
    test_pair_is_pos = test_pairs["y"].eq(1)
    test_pos_pairs = known_pairs[test_pair_is_pos]
    test_neg_pairs = known_pairs[~test_pair_is_pos]
    matrix["is_known_positive"] = create_flag_column(test_pos_pairs)
    matrix["is_known_negative"] = create_flag_column(test_neg_pairs)

    # Flag clinical trials data

    return


def generate_report(
    data: pd.DataFrame,
    n_reporting: int,
    drugs: pd.DataFrame,
    diseases: pd.DataFrame,
    known_pairs: pd.DataFrame,
    score_col_name: str,
) -> pd.DataFrame:
    """Generates a report with the top pairs.

    Args:
        data: Pairs dataset.
        n_reporting: Number of pairs in the report.
        drugs: Dataframe containing names and IDs for the list of drugs.
        diseases: Dataframe containing names and IDs for the list of diseases.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        score_col_name: Probability score column name.

    Returns:
        Dataframe containing the top pairs with additional information for the drugs and diseases.
    """
    # Select the top n_reporting rows
    top_pairs = data.head(n_reporting).copy()

    # Generate curie to name dictionaries
    drug_curie_to_name = {row["curie"]: row["name"] for _, row in drugs.iterrows()}
    disease_curie_to_name = {
        row["curie"]: row["name"] for _, row in diseases.iterrows()
    }

    # Add additional information for drugs and diseases
    top_pairs["drug_name"] = top_pairs["source"].map(drug_curie_to_name)
    top_pairs["disease_name"] = top_pairs["target"].map(disease_curie_to_name)

    # Rename ID columns
    top_pairs = top_pairs.rename(columns={"source": "drug_id"})
    top_pairs = top_pairs.rename(columns={"target": "disease_id"})

    # Reorder columns for better readability
    columns_order = [
        "drug_id",
        "drug_name",
        "disease_id",
        "disease_name",
        score_col_name,
        "is_known_positive",
        "is_known_negative",
    ]
    top_pairs = top_pairs[columns_order]

    return top_pairs
