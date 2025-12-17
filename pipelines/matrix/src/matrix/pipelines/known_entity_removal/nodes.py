import logging
from functools import reduce

import pandas as pd
import pyspark.sql as ps
from matrix_inject.inject import inject_object
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .evaluation import give_projected_proportion, give_removal_rate, give_retrieval_rate  # noqa: F401
from .mondo_ontology import OntologyMONDO

logger = logging.getLogger(__name__)

LABEL_COLS = {
    "reached_sac": Column(bool, nullable=False),
    "reached_deep_dive": Column(bool, nullable=False),
    "reached_med_review": Column(bool, nullable=False),
    "reached_triage": Column(bool, nullable=False),
    "archived_known_on_label": Column(bool, nullable=False),
    "archived_known_off_label": Column(bool, nullable=False),
    "archived_known_entity": Column(bool, nullable=False),
    "triaged_not_known_entity": Column(bool, nullable=False),
}


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug_id": Column(str, nullable=False),
            "disease_id": Column(str, nullable=False),
        },
        unique=["drug_id", "disease_id"],
    )
)
def concatenate_datasets(
    datasets_to_include: dict[str, dict[str, bool]], **all_datasets: dict[str, ps.DataFrame]
) -> ps.DataFrame:
    """
    Concatenate datasets based the inclusion parameters.

    datasets_to_include: For each available dataset, contains a dictionary of boolean values for each pair type
        (positives, negatives) determining whether to include positive (y = 1) and/or negative (y = 0) pairs.
    all_datasets: A dictionary of all available datasets.
        Columns required: "subject", "object", "y"

    Returns:
        A dataframe with the unique drug-disease pairs from the union of the included datasets.
    """
    y_values_required = {
        dataset_name: [
            y
            for y, pair_type in zip([0, 1], ["negatives", "positives"])
            if datasets_to_include[dataset_name][pair_type]
        ]
        for dataset_name in all_datasets.keys()
    }

    dataframes_to_concatenate = [
        df.filter(F.col("y").isin(y_values_required[dataset_name])).select(
            F.col("subject").alias("drug_id"), F.col("object").alias("disease_id")
        )
        for dataset_name, df in all_datasets.items()
        if len(y_values_required[dataset_name]) > 0
    ]
    return reduce(lambda df1, df2: df1.union(df2), dataframes_to_concatenate).distinct()


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug_id": Column(str, nullable=False),
            "disease_id": Column(str, nullable=False),
        },
        unique=["drug_id", "disease_id"],
    )
)
@inject_object()
def apply_mondo_expansion(
    mondo_ontology: OntologyMONDO,
    concatenated_ground_truth: ps.DataFrame,
) -> ps.DataFrame:
    """
    Apply Mondo ontology expansion to the concatenated ground truth.
    """
    # We collect unique disease and apply get_equivalent_mondo_ids in a single pass
    # to avoid distributing non-serializable class OntologyMONDO to workers.
    spark = SparkSession.builder.getOrCreate()
    unique_diseases = [x.disease_id for x in concatenated_ground_truth.select("disease_id").distinct().collect()]
    equivalent_diseases = spark.createDataFrame(
        [(id, mondo_ontology.get_equivalent_mondo_ids(id) + [id]) for id in unique_diseases],
        schema=["disease_id", "equivalent_disease_id"],
    ).withColumn("equivalent_disease_id", F.explode("equivalent_disease_id"))

    return (
        concatenated_ground_truth.join(equivalent_diseases, on="disease_id", how="left")
        .select("drug_id", F.col("equivalent_disease_id").alias("disease_id"))
        .distinct()
    )


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug_translator_id": Column(str, nullable=False),
            "ec_drug_id": Column(str, nullable=False),
            "target": Column(str, nullable=False),
            "is_known_entity": Column(bool, nullable=False),
        },
        unique=["ec_drug_id", "target"],
    )
)
def create_known_entity_matrix(
    drug_list: ps.DataFrame,
    disease_list: ps.DataFrame,
    expanded_ground_truth: ps.DataFrame,
) -> ps.DataFrame:
    """
    Create the known entity matrix in accordance with output schema expected by Orchard.

    FUTURE: Consider tracking upstream source for known entities.
    """
    return (
        drug_list.select(F.col("ec_id").alias("ec_drug_id"), F.col("id").alias("drug_translator_id"))
        .join(disease_list.select(F.col("core_id").alias("target")), how="cross")
        .join(
            expanded_ground_truth.select(
                F.col("drug_id").alias("drug_translator_id"), F.col("disease_id").alias("target")
            ).withColumn("is_known_entity", F.lit(True)),
            on=["drug_translator_id", "target"],
            how="left",
        )
        .fillna(False, subset=["is_known_entity"])
    )


def _remove_null_names(orchard_pairs: ps.DataFrame) -> ps.DataFrame:
    """
    Remove Orchard pairs with null names.
    """
    return orchard_pairs[orchard_pairs["drug_name"].notna() & orchard_pairs["disease_name"].notna()]


def _convert_timestamps_to_datetime(orchard_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp columns to datetime.
    """
    # orchard_pairs["last_created_at_at_report_date"] values expected to be pd.Timestamp.
    # We convert to pd.Timestamp format to also deal with string format from the fabricator.
    timestamp_columns = ["report_date", "last_created_at_at_report_date"]
    orchard_pairs[timestamp_columns] = orchard_pairs[timestamp_columns].map(lambda x: pd.Timestamp(x).to_pydatetime())
    return orchard_pairs


def _restrict_to_report_date(orchard_pairs: pd.DataFrame, orchard_report_date: str) -> dict[str, pd.DataFrame]:
    """Restrict "Orchard pairs by month" to specified report date. Ensures reproducibility.

    Args:
        orchard_pairs: "Orchard pairs by month" dataframe.
        orchard_report_date: Report date to restrict the Orchard data to.
            Format: "YYYY-MM", or "latest" (to use the latest available pairs).

    Returns:
        Tuple:
            - Restricted pairs dataframe with latest status for the report date.
            - Report date as a dictionary.
    """
    if orchard_report_date == "latest":
        orchard_report_date = orchard_pairs["report_date"].max()
    else:
        orchard_report_date = pd.Timestamp(orchard_report_date).to_pydatetime()
    restricted_orchard_pairs = orchard_pairs[orchard_pairs["report_date"] == orchard_report_date]
    restricted_orchard_pairs["last_created_at"] = restricted_orchard_pairs["last_created_at_at_report_date"].map(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
    )
    restricted_orchard_pairs = restricted_orchard_pairs.drop(columns=["report_date", "last_created_at_at_report_date"])

    report_date_info = pd.DataFrame({"orchard_data_report_date": [orchard_report_date]})
    return restricted_orchard_pairs, report_date_info


def _add_labels(orchard_pairs: ps.DataFrame) -> ps.DataFrame:
    """
    Add status and archival labels for Orchard pairs.
    """
    # Add labels for whether a pair reached a given status
    orchard_pairs["reached_sac"] = orchard_pairs["status_transitions_up_to_report_date"].str.contains("SAC_ENDORSED")
    orchard_pairs["reached_deep_dive"] = orchard_pairs["status_transitions_up_to_report_date"].str.contains("DEEP_DIVE")
    orchard_pairs["reached_med_review"] = orchard_pairs["status_transitions_up_to_report_date"].str.contains(
        "MEDICAL_REVIEW"
    )
    orchard_pairs["reached_triage"] = orchard_pairs["status_transitions_up_to_report_date"].str.contains("TRIAGE")

    # Add labels for known entity pairs
    orchard_pairs["archived_known_on_label"] = (
        orchard_pairs["depriortization_reason_at_report_date"] == "DRUG_ON_LABEL_FOR_DISEASE"
    )
    orchard_pairs["archived_known_off_label"] = (
        orchard_pairs["depriortization_reason_at_report_date"] == "DRUG_WIDELY_USED_OFF_LABEL"
    )
    orchard_pairs["archived_known_entity"] = (
        orchard_pairs["archived_known_on_label"] | orchard_pairs["archived_known_off_label"]
    )

    # Add labels for positive pairs : triaged and not archived as known entity
    orchard_pairs["triaged_not_known_entity"] = (orchard_pairs["reached_triage"]) & (
        ~orchard_pairs["archived_known_entity"]
    )

    return orchard_pairs


@check_output(
    schema=DataFrameSchema(
        columns={
            # "drug_name": Column(str, nullable=False),
            # "disease_name": Column(str, nullable=False),
            "drug_id": Column(str, nullable=False),
            "disease_id": Column(str, nullable=False),
            "last_created_at": Column(str, nullable=False),
            **LABEL_COLS,
        },
        unique=["drug_id", "disease_id"],
    ),
    df_name="processed_orchard_pairs",
)
def preprocess_orchard_pairs(orchard_pairs: pd.DataFrame, orchard_report_date: str) -> dict[str, pd.DataFrame]:
    """
    Preprocess the "pairs_status_transitions_by_month" dataset.
    """
    # Record columns before processing
    old_columns = orchard_pairs.columns

    # Process orchard pairs
    # orchard_pairs = _remove_null_names(orchard_pairs)
    orchard_pairs = _convert_timestamps_to_datetime(orchard_pairs)
    orchard_pairs, report_date_info = _restrict_to_report_date(orchard_pairs, orchard_report_date)
    orchard_pairs = _add_labels(orchard_pairs)
    orchard_pairs = orchard_pairs.rename(columns={"drug_kg_node_id": "drug_id", "disease_kg_node_id": "disease_id"})

    # Select newly created or renamed columns and drug/disease name columns # TODO update if needed
    new_columns = orchard_pairs.columns
    columns_to_keep = [
        col for col in new_columns if col not in old_columns
    ]  # ["drug_name", "disease_name"] + [col for col in new_columns if col not in old_columns]
    return {"processed_orchard_pairs": orchard_pairs[columns_to_keep], "report_date_info": report_date_info}


@check_output(
    schema=DataFrameSchema(
        columns={
            # "drug_name": Column(str, nullable=False),
            # "disease_name": Column(str, nullable=False),
            "drug_id": Column(str, nullable=False),
            "disease_id": Column(str, nullable=False),
            "last_created_at": Column(str, nullable=False),
            **LABEL_COLS,
            "is_known_entity": Column(bool, nullable=True),
        },
        unique=["drug_id", "disease_id"],
    )
)
def add_predictions_column(
    orchard_pairs: ps.DataFrame,
    known_entity_matrix: ps.DataFrame,
) -> ps.DataFrame:
    """Add a column with the known entity predictions.

    NOTE: Any Orchard pair not in the predictions dataframe is marked as "not known entity" by default.

    Args:
        orchard_pairs: Preprocessed Orchard pairs dataframe.
        known_entity_matrix: Dataframe with known entity predictions across all drugs and diseases.
    """
    orchard_pairs_with_preds = orchard_pairs.join(
        known_entity_matrix.select(
            F.col("ec_drug_id").alias("drug_id"), F.col("target").alias("disease_id"), F.col("is_known_entity")
        ),
        on=["drug_id", "disease_id"],
        how="left",
    )

    num_nulls = orchard_pairs_with_preds.filter(F.col("is_known_entity").isNull()).count()
    logger.warning(f"Number of Orchard pairs without known entity predictions: {num_nulls}")

    return orchard_pairs_with_preds.fillna(False, subset=["is_known_entity"])


def calculate_metrics(
    orchard_pairs_with_preds: ps.DataFrame, metrics_to_report: dict[str, dict[str, str]]
) -> dict[str, float]:
    """Calculate the metrics for the known entity removal pipeline.

    Args:
        orchard_pairs_with_preds: Dataframe with known entity predictions and labels.
        metrics_to_report: Dictionary of metrics to report.
    """
    metrics = {}
    for metric_name, metric_info in metrics_to_report.items():
        evaluation_fn = eval(metric_info["evaluation_fn"])
        label_col_name = metric_info["label_col_name"]
        mean, std = evaluation_fn(orchard_pairs_with_preds, label_col_name, filter_col_name="is_known_entity")
        metrics[metric_name + "_mean"] = mean
        metrics[metric_name + "_std"] = std
    return metrics
