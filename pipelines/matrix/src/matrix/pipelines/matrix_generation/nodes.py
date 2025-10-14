import logging
from typing import Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from matplotlib.figure import Figure
from matrix.datasets.graph import KnowledgeGraph
from matrix.pipelines.matrix_generation.reporting_plots import ReportingPlotGenerator
from matrix.pipelines.matrix_generation.reporting_tables import ReportingTableGenerator
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.preprocessing_model import ModelWithTransformers
from matrix_inject.inject import inject_object
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql.types import DoubleType, StructField, StructType
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
    matrix: pd.DataFrame,
    known_pairs: pd.DataFrame,
    clinical_trials: Optional[pd.DataFrame] = None,
    off_label: Optional[pd.DataFrame] = None,
    orchard: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Adds boolean columns flagging known positives and known negatives.

    Args:
        matrix: Drug-disease pairs dataset.
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.
        off_label: Pairs dataset representing off label usage.
        orchard: Pairs dataset representing orchard feedback data.

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

    # Flag off label data
    off_label = off_label.rename(columns={"subject": "source", "object": "target"})
    matrix["off_label"] = create_flag_column(off_label)  # all pairs are positive

    # Flag orchard data if available
    if orchard is not None:
        orchard = orchard.rename(columns={"subject": "source", "object": "target"})
        matrix["high_evidence_matrix"] = create_flag_column(orchard[orchard["high_evidence_matrix"] == 1])
        matrix["mid_evidence_matrix"] = create_flag_column(orchard[orchard["mid_evidence_matrix"] == 1])
        matrix["high_evidence_crowdsourced"] = create_flag_column(orchard[orchard["high_evidence_crowdsourced"] == 1])
        matrix["mid_evidence_crowdsourced"] = create_flag_column(orchard[orchard["mid_evidence_crowdsourced"] == 1])
        matrix["archive_biomedical_review"] = create_flag_column(orchard[orchard["archive_biomedical_review"] == 1])

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
    off_label: Optional[pd.DataFrame] = None,
    orchard: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Function to generate matrix dataset.

    FUTURE: Consider rewriting operations in PySpark for speed

    Args:
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph: Object containing node embeddings.
        clinical_trials: Pairs dataset representing outcomes of recent clinical trials.
        off_label: Pairs dataset representing off label drug disease uses.
        orchard: Pairs dataset representing orchard feedback data.

    Returns:
        Pairs dataframe containing all combinations of drugs and diseases that do not lie in the training set.
    """
    # Collect list of drugs and diseases
    diseases_lst = diseases["id"].tolist()
    # This try/except is to make modelling_run pipeline compatible with old drug list (pre-migration)
    try:
        drugs_df = drugs[["id", "ec_id"]]
        column_remapping = {"ec_id": "ec_drug_id", "id": "source"}
    except KeyError:
        logger.warning("ec_id column not found in drugs dataframe; using id column instead")
        column_remapping = {"id": "source"}
        drugs_df = drugs[["id"]]

    # Remove duplicates
    drugs_df = drugs_df.drop_duplicates()
    diseases_lst = list(set(diseases_lst))
    # Remove drugs and diseases without embeddings
    nodes_with_embeddings = set(graph._nodes["id"])
    drugs_df = drugs_df[drugs_df["id"].isin(nodes_with_embeddings)]
    diseases_lst = [disease for disease in diseases_lst if disease in nodes_with_embeddings]

    # Generate all combinations
    matrix_slices = []
    for disease in tqdm(diseases_lst):
        matrix_slice = pd.DataFrame(drugs_df.rename(column_remapping).assign(target=disease))
        matrix_slices.append(matrix_slice)

    # Concatenate all slices at once
    matrix = pd.concat(matrix_slices, ignore_index=True)

    # Remove training set
    train_pairs = known_pairs[~known_pairs["split"].eq("TEST")]
    train_pairs_set = set(zip(train_pairs["source"], train_pairs["target"]))
    is_in_train = matrix.apply(lambda row: (row["source"], row["target"]) in train_pairs_set, axis=1)
    matrix = matrix[~is_in_train]
    # Add flag columns for known positives and negatives
    matrix = _add_flag_columns(matrix, known_pairs, clinical_trials, off_label, orchard)
    return matrix


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(T.StringType(), nullable=False),
            "target": Column(T.StringType(), nullable=False),
            # The three score columns are passed as parameters of the function
            "not treat score": Column(T.DoubleType(), nullable=False),
            "treat score": Column(T.DoubleType(), nullable=False),
            "unknown score": Column(T.DoubleType(), nullable=False),
            "rank": Column(T.LongType(), nullable=False),
            "quantile_rank": Column(T.DoubleType(), nullable=False),
        },
        unique=["source", "target"],
    )
)
def make_predictions_and_sort(
    node_embeddings: ps.DataFrame,
    pairs: ps.DataFrame,
    treat_score_col_name: str,
    not_treat_score_col_name: str,
    unknown_score_col_name: str,
    model: ModelWrapper,
) -> ps.DataFrame:
    """Generate and sort probability scores for a drug-disease dataset.

    Args:
        node_embeddings: Dataframe with node embeddings.
        pairs: Drug-disease pairs to predict scores for.
        treat_score_col_name: Name of the column for treatment scores.
        not_treat_score_col_name: Name of the column for non-treatment scores.
        unknown_score_col_name: Name of the column for unknown scores.
        model: Ensemble model capable of producing probability scores.

    Returns:
        Pairs dataset sorted by score with their rank and quantile rank.
    """

    embeddings = node_embeddings.select("id", "topological_embedding")

    pairs_with_embeddings = (
        # TODO: remnant from pyarrow/pandas conversion, find in which node it is created
        pairs.drop("__index_level_0__")
        .join(
            embeddings.withColumnsRenamed({"id": "target", "topological_embedding": "target_embedding"}),
            on="target",
            how="left",
        )
        .join(
            embeddings.withColumnsRenamed({"id": "source", "topological_embedding": "source_embedding"}),
            on="source",
            how="left",
        )
        .filter(F.col("source_embedding").isNotNull() & F.col("target_embedding").isNotNull())
    )

    def model_predict(partition_df: pd.DataFrame) -> pd.DataFrame:
        model_predictions = model.predict_proba(partition_df)

        # Assign averaged predictions to columns
        partition_df[not_treat_score_col_name] = model_predictions[:, 0]
        partition_df[treat_score_col_name] = model_predictions[:, 1]
        partition_df[unknown_score_col_name] = model_predictions[:, 2]

        return partition_df.drop(columns=["source_embedding", "target_embedding"])

    structfields_to_keep = [
        col for col in pairs_with_embeddings.schema if col.name not in ["target_embedding", "source_embedding"]
    ]
    model_predict_schema = StructType(
        structfields_to_keep
        + [
            StructField(not_treat_score_col_name, DoubleType(), True),
            StructField(treat_score_col_name, DoubleType(), True),
            StructField(unknown_score_col_name, DoubleType(), True),
        ]
    )

    pairs_with_scores = pairs_with_embeddings.groupBy("target").applyInPandas(model_predict, model_predict_schema)

    pairs_sorted = pairs_with_scores.orderBy(treat_score_col_name, ascending=False)

    # We are using the RDD.zipWithIndex function here. Getting it through the DataFrame API would involve a Window function without partition, effectively pulling all data into one single partition.
    # Here is what happens in the next line:
    # 1. zipWithIndex creates a tuple with the shape (row, index)
    # 2. When moving from RDD to DataFrame, the column names are named after the Scala tuple fields: _1 for the row and _2 for the index
    # 3. We're adding 1 to the rank so that it is not zero indexed
    pairs_ranked = pairs_sorted.rdd.zipWithIndex().toDF().select(F.col("_1.*"), (F.col("_2") + 1).alias("rank"))

    pairs_ranked_count = pairs_ranked.count()
    return pairs_ranked.withColumn("quantile_rank", F.col("rank") / pairs_ranked_count)


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


def package_model_with_transformers(
    transformers: dict,
    model: ModelWrapper,
    features: list[str],
) -> ModelWrapper:
    """Bundle transformers, features, and model into a single callable object."""
    return ModelWithTransformers(model, transformers, features)
