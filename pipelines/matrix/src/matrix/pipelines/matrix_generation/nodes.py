import functools
import itertools
import logging
import operator
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

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
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, BooleanType, FloatType, StringType
from pyspark.sql.window import Window
from sklearn.impute._base import _BaseImputer
from tqdm import tqdm

logger = logging.getLogger(__name__)


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(StringType(), nullable=False),
            "topological_embedding": Column(ArrayType(FloatType()), nullable=False),
            "is_drug": Column(BooleanType(), nullable=False),
            "is_disease": Column(BooleanType(), nullable=False),
        },
        unique=["id"],
    )
)
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
        .fillna(False, subset=("is_drug", "is_disease"))
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
    graph: ps.DataFrame,
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
    # NOTE: Ensures we load only the id column before moving to pandas world
    nodes_with_embeddings = set(graph.select("id").toPandas()["id"].tolist())
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


def make_predictions_and_sort(
    graph: KnowledgeGraph,
    data: ps.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
) -> pd.DataFrame:
    """Generate probability scores for drug-disease dataset.

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """
    data = data.drop("__index_level_0__")  # remnant from pyarrow/pandas conversion, not sure in which previous node
    embeddings = graph.select("id", "topological_embedding")

    data = data.join(
        embeddings.withColumnsRenamed({"id": "source", "topological_embedding": "source_embedding"}),
        on="source",
        how="left",
    ).join(
        embeddings.withColumnsRenamed({"id": "target", "topological_embedding": "target_embedding"}),
        on="target",
        how="left",
    )

    # Drop rows without source/target embeddings
    # data = drop_rows_with_empty_feature_values(data, transformers)

    def predict_partition(partitionindex: int, partition: Iterable[Row]) -> Iterator[Row]:
        partition_df = pd.DataFrame.from_records(row.asDict() for row in partition)
        if partition_df.empty:
            logger.warning(f"partition with index {partitionindex} is empty")
            return

        transformed = apply_transformers(partition_df, transformers)
        batch_features = _extract_elements_in_list(transformed.columns, features, True)

        predictions = model.predict_proba(transformed[batch_features].values)
        predictions_df = pd.DataFrame(
            predictions, columns=[not_treat_score_col_name, treat_score_col_name, unknown_score_col_name]
        )

        result_df = pd.concat([partition_df, predictions_df], axis=1)
        for row in result_df.to_dict("records"):
            yield Row(**row)

    # TODO: experiment with Spark vectorized UDFs instead of using the RDD API, it should provide better performance
    data = data.rdd.mapPartitionsWithIndex(predict_partition).toDF()
    # Sort by treat score in descending order
    sorted_data = data.orderBy(F.col(treat_score_col_name).desc())
    # Add rank and quantile rank columns using window functions
    window_spec = Window.orderBy(F.col(treat_score_col_name).desc())
    sorted_data = sorted_data.withColumn("rank", F.row_number().over(window_spec))
    total_rows = sorted_data.count()
    sorted_data = sorted_data.withColumn("quantile_rank", F.col("rank") / F.lit(total_rows))
    return sorted_data


def drop_rows_with_empty_feature_values(data: ps.DataFrame, transformers):
    # Retrieve rows where feature columns are null
    # NOTE: This only happens in a rare scenario where the node synonymizer
    # provided an identifier for a node that does _not_ exist in our KG.
    # https://github.com/everycure-org/matrix/issues/409
    features: list[str] = list(itertools.chain.from_iterable(x["features"] for x in transformers.values()))

    if logger.isEnabledFor(logging.INFO):
        logging.info(f"Checking for dropped pairs because one of the features ({features}) is empty…")
        removed = (
            data.filter(functools.reduce(operator.or_, (F.col(colname).isNull() for colname in features)))
            .select("source", "target")
            .cache()
        )
        removed_pairs = removed.take(50)  # Can be extended, but mind OOM.
        if removed_pairs:
            logger.warning(f"Dropped {removed.count()} pairs during generation!")
            logger.warning("Dropped (subset of 50 shown): %s", ", ".join(f"({p[0]}, {p[1]})" for p in removed_pairs))
        removed.unpersist()

    return data.dropna(subset=features, how="any")


@inject_object()
def generate_reports(
    sorted_matrix_df: pd.DataFrame | ps.DataFrame,
    strategies: dict[str, ReportingPlotGenerator | ReportingTableGenerator],
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
    names = [strategy.name for strategy in strategies.values()]
    if len(names) != len(set(names)):
        raise ValueError("Reporting strategy names must be unique.")

    reports_dict = {}
    for strategy in strategies.values():
        reports_dict[strategy.name] = strategy.generate(sorted_matrix_df, **kwargs)
    return reports_dict
