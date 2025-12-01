import logging
from typing import Optional

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from matplotlib.figure import Figure
from matrix.pipelines.matrix_generation.reporting_plots import ReportingPlotGenerator
from matrix.pipelines.matrix_generation.reporting_tables import ReportingTableGenerator
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.preprocessing_model import ModelWithTransformers
from matrix_inject.inject import inject_object
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql.types import DoubleType, StructField, StructType

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
        drugs.select("id")
        .withColumn("is_drug", F.lit(True))
        .unionByName(diseases.select("id").withColumn("is_disease", F.lit(True)), allowMissingColumns=True)
        .join(nodes, on="id", how="inner")
        .select("is_drug", "is_disease", "id", "topological_embedding")
        .withColumn("is_drug", F.coalesce(F.col("is_drug"), F.lit(False)))
        .withColumn("is_disease", F.coalesce(F.col("is_disease"), F.lit(False)))
    )


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            # "source_ec_id": Column(str, nullable=False), # Column present only if drugs dataframe has ec_id column
            "target": Column(str, nullable=False),
            "is_known_positive": Column(bool, nullable=False),
            "is_known_negative": Column(bool, nullable=False),
            "trial_sig_better": Column(bool, nullable=False),
            "trial_non_sig_better": Column(bool, nullable=False),
            "trial_sig_worse": Column(bool, nullable=False),
            "trial_non_sig_worse": Column(bool, nullable=False),
            "off_label": Column(bool, nullable=False),
        },
        unique=["source", "target"],
    )
)
@inject_object()
def generate_pairs(
    known_pairs: ps.DataFrame,
    drugs: ps.DataFrame,
    diseases: ps.DataFrame,
    node_embeddings: ps.DataFrame,
    clinical_trials: Optional[ps.DataFrame] = None,
    off_label: Optional[ps.DataFrame] = None,
    orchard: Optional[ps.DataFrame] = None,
) -> ps.DataFrame:
    """Function to generate matrix dataset.

    Args:
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        drugs: Dataframe containing IDs for the list of drugs.
        diseases: Dataframe containing IDs for the list of diseases.
        graph_nodes: pyspark dataframe containing node embeddings.
        clinical_trials: pyspark dataframe representing outcomes of recent clinical trials.
        off_label: pyspark dataframe representing off label drug disease uses.
        orchard: pyspark dataframe representing orchard feedback data.

    Returns:
        pyspark dataframe containing all combinations of drugs and diseases that do not lie in the training set with flags
    """

    def get_join_columns(left_columns: list[str], right_columns: list[str]) -> list[str]:
        if "source_ec_id" in left_columns and "source_ec_id" in right_columns:
            return ["source_ec_id", "target"]
        else:
            return ["source", "target"]

    # 1. Filter out drugs and diseases without embeddings
    drugs_in_graph = drugs.alias("drugs").join(node_embeddings, on="id", how="inner")
    if "ec_id" in drugs.columns:
        drugs_in_graph = drugs_in_graph.select(
            F.col("drugs.id").alias("source"), F.col("drugs.ec_id").alias("source_ec_id")
        )
    else:
        drugs_in_graph = drugs_in_graph.select(F.col("drugs.id").alias("source"))

    diseases_in_graph = (
        diseases.alias("diseases")
        .join(node_embeddings, on="id", how="inner")
        .select(F.col("diseases.id").alias("target"))
    )

    # 2. Generate all drug / disease combinations that are not in the training set
    matrix = drugs_in_graph.crossJoin(diseases_in_graph)

    matrix_known_pairs_join_columns = get_join_columns(matrix.columns, known_pairs.columns)
    train_pairs = known_pairs.filter(F.col("split") == "TRAIN")
    matrix = matrix.join(train_pairs, on=matrix_known_pairs_join_columns, how="leftanti")

    # 3. Add known pairs flags
    test_pairs = known_pairs.filter(F.col("split") == "TEST")
    matrix = (
        matrix.alias("matrix")
        .join(test_pairs.alias("test_pairs"), on=matrix_known_pairs_join_columns, how="left")
        .select(
            F.col("matrix.*"),
            (F.coalesce(F.col("test_pairs.y") == 1, F.lit(False))).alias("is_known_positive"),
            (F.coalesce(F.col("test_pairs.y") != 1, F.lit(False))).alias("is_known_negative"),
        )
    )

    # 4. Add clinical trials flags
    if clinical_trials is not None:
        matrix_clinical_trials_join_columns = get_join_columns(matrix.columns, clinical_trials.columns)
        clinical_trials = clinical_trials.withColumnsRenamed({"subject": "source", "object": "target"})
        matrix = (
            matrix.alias("matrix")
            .join(clinical_trials.alias("clinical_trials"), on=matrix_clinical_trials_join_columns, how="left")
            .select(
                F.col("matrix.*"),
                (F.coalesce(F.col("clinical_trials.significantly_better") == 1, F.lit(False))).alias(
                    "trial_sig_better"
                ),
                (F.coalesce(F.col("clinical_trials.non_significantly_better") == 1, F.lit(False))).alias(
                    "trial_non_sig_better"
                ),
                (F.coalesce(F.col("clinical_trials.significantly_worse") == 1, F.lit(False))).alias("trial_sig_worse"),
                (F.coalesce(F.col("clinical_trials.non_significantly_worse") == 1, F.lit(False))).alias(
                    "trial_non_sig_worse"
                ),
            )
        )

    # 5. Add off label flags
    if off_label is not None:
        matrix_off_label_join_columns = get_join_columns(matrix.columns, off_label.columns)
        off_label = off_label.withColumnsRenamed(
            colsMap={"subject": "source", "subject_ec_id": "source_ec_id", "object": "target"}
        )
        matrix = (
            matrix.alias("matrix")
            .join(off_label.alias("off_label"), on=matrix_off_label_join_columns, how="left")
            .select(
                F.col("matrix.*"),
                (F.coalesce(F.col("off_label.off_label") == 1, F.lit(False))).alias("off_label"),
            )
        )

    # 6. Add orchard flags
    if orchard is not None:
        matrix_orchard_join_columns = get_join_columns(matrix.columns, orchard.columns)
        orchard = orchard.withColumnsRenamed({"subject": "source", "object": "target"})
        matrix = (
            matrix.alias("matrix")
            .join(orchard.alias("orchard"), on=matrix_orchard_join_columns, how="left")
            .select(
                F.col("matrix.*"),
                (F.coalesce(F.col("orchard.high_evidence_matrix") == 1, F.lit(False))).alias("high_evidence_matrix"),
                (F.coalesce(F.col("orchard.mid_evidence_matrix") == 1, F.lit(False))).alias("mid_evidence_matrix"),
                (F.coalesce(F.col("orchard.high_evidence_crowdsourced") == 1, F.lit(False))).alias(
                    "high_evidence_crowdsourced"
                ),
                (F.coalesce(F.col("orchard.mid_evidence_crowdsourced") == 1, F.lit(False))).alias(
                    "mid_evidence_crowdsourced"
                ),
                (F.coalesce(F.col("orchard.archive_biomedical_review") == 1, F.lit(False))).alias(
                    "archive_biomedical_review"
                ),
            )
        )
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
