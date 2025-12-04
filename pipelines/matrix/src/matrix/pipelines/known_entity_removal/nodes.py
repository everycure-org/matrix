from functools import reduce

import pyspark.sql as ps
from matrix_inject.inject import inject_object
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .mondo_ontology import OntologyMONDO


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
