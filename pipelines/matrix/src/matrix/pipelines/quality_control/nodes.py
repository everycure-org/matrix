import os
import shutil
import pyspark as ps
import pandas as pd
from typing import List
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from refit.v1.core.inject import inject_object
from pyspark.sql import types as T

from .quality_control import QaulityControl

"""
Sprint Goal
Implement tracking for following metrics:
[x] # of nodes ingested
[x] # of nodes normalized (via Node Normalizer)
[x] # of nodes that failed normalization
[x] Node to Normalized Node map -> this already exists: integration.int.rtx.nodes_norm_mapping
[x] # of edges ingested
[ ] # of edges that are valid (Biolink predicate with subject, object, knowledge level, agent type, and primary knowledge source)
[ ] # of edges not valid
[ ] Log of invalid edges and why they were flagged as invalid
[x] Count of node types and edge types
[ ] Track node-edge-node relationships (count Subject-Predicate-Object triples)
[ ] Identify unconnected nodes and nodes removed post-normalization

"""


@inject_object()
def run_quality_control(df: DataFrame, controls: List[QaulityControl]) -> DataFrame:
    """Function to run given quality controls on input dataframe."""

    # Initialise spark
    spark = SparkSession.builder.getOrCreate()

    # Create an empty DataFrame with the schema
    result: DataFrame = spark.createDataFrame(
        [],
        schema=T.StructType(
            [
                T.StructField("metric", T.StringType(), True),
                T.StructField("value", T.IntegerType(), True),  # TODO: Refine type
            ]
        ),
    )

    # Run each control suite on the given dataframe and union
    # the results together
    # TODO: Namespace the metric to add the QualityControl name as a prefix?
    for control in controls:
        result = result.union(control.run(df))

    return result


# TODO: Break functions below into logical suites to allow resuability
# TODO: Ensure catalog is _ALWAYS_ used, no reading/writing directly


def write_single_parquet(df: DataFrame, output_path: str) -> None:
    """Write a DataFrame to a single Parquet file."""
    temp_dir = output_path + "_temp"
    df.coalesce(1).write.mode("overwrite").parquet(temp_dir)

    for file in os.listdir(temp_dir):
        if file.endswith(".parquet"):
            shutil.move(os.path.join(temp_dir, file), output_path)
            break

    shutil.rmtree(temp_dir)


def count_categories(nodes: ps.sql.DataFrame, column_name: str) -> list:
    """Count nodes by category."""
    counts = nodes.groupBy(column_name).count()
    return counts.toPandas().to_dict("records")


def count_success_fail(norm_map: ps.sql.DataFrame) -> dict:
    """Count normalization successes and failures."""
    counts = norm_map.groupBy("normalization_success").count()
    return {row["normalization_success"]: row["count"] for row in counts.collect()}


def group_by_prefix(norm_map: ps.sql.DataFrame, success: bool) -> list:
    """Group normalized nodes by prefix and success."""
    grouped = (
        norm_map.filter(F.col("normalization_success") == success)
        .withColumn("prefix", F.split(F.col("normalized_id"), ":")[0])
        .groupBy("prefix")
        .count()
    )
    return grouped.toPandas().to_dict("records")


def ingestion(
    nodes: ps.sql.DataFrame,
    edges: ps.sql.DataFrame,
    dataset_name: str,
    output_path: str,
) -> None:
    nodes_count = nodes.count()
    edges_count = edges.count()

    node_prefix = nodes.withColumn("prefix", F.split(F.col("id:ID"), ":")[0])
    prefix_counts = node_prefix.groupBy("prefix").count().toPandas().to_dict("records")

    summary_data = pd.DataFrame(
        [
            {"metric": f"total number of nodes in {dataset_name}", "value": nodes_count},
            {"metric": f"number of edges {dataset_name}", "value": edges_count},
            {"metric": f"number of unique CURIE prefixes {dataset_name}", "value": len(prefix_counts)},
            {"metric": f"prefix counts {dataset_name}", "value": str(prefix_counts)},
        ]
    )

    spark = ps.sql.SparkSession.builder.getOrCreate()
    write_single_parquet(spark.createDataFrame(summary_data), output_path)


def integration(
    nodes: ps.sql.DataFrame,
    nodes_transformed: ps.sql.DataFrame,
    norm_nodes: ps.sql.DataFrame,
    norm_nodes_map: ps.sql.DataFrame,
    dataset_name: str,
    output_path: str,
) -> None:
    ingested_count = nodes.count()
    transformed_count = nodes_transformed.count()
    normalized_count = norm_nodes.count()
    # difference_transformed = ingested_count - transformed_count
    difference_normed = ingested_count - normalized_count

    category_counts_transformed = count_categories(nodes_transformed, "category")
    success_fail_counts = count_success_fail(norm_nodes_map)
    success_count = success_fail_counts.get(True, 0)
    fail_count = success_fail_counts.get(False, 0)

    true_prefix_counts = group_by_prefix(norm_nodes_map, True)
    false_prefix_counts = group_by_prefix(norm_nodes_map, False)

    norm_category_counts = count_categories(norm_nodes, "category")
    failed_category_counts = count_categories(
        nodes_transformed.join(norm_nodes_map, "id").filter(~F.col("normalization_success")),
        "category",
    )

    difference_match = "Yes" if fail_count == difference_normed else "No"

    summary_data = pd.DataFrame(
        [
            {"metric": f"total ingested nodes in {dataset_name}", "value": ingested_count},
            {"metric": f"total transformed nodes in {dataset_name}", "value": transformed_count},
            {
                "metric": f"node category counts after transformation in {dataset_name}",
                "value": str(category_counts_transformed),
            },
            {"metric": f"total normalized nodes in {dataset_name}", "value": normalized_count},
            {"metric": f"normalization success count in {dataset_name}", "value": success_count},
            {"metric": f"normalization failure count in {dataset_name}", "value": fail_count},
            {"metric": f"does 'FALSE' count match difference in {dataset_name}", "value": difference_match},
            {
                "metric": f"node category counts after normalization in {dataset_name}",
                "value": str(norm_category_counts),
            },
            {
                "metric": f"node category counts that failed normalization in {dataset_name}",
                "value": str(failed_category_counts),
            },
            {"metric": f"Normalization success CURIE counts in {dataset_name}", "value": str(true_prefix_counts)},
            {"metric": f"Normalization failure CURIE counts in {dataset_name}", "value": str(false_prefix_counts)},
        ]
    )

    spark = ps.sql.SparkSession.builder.getOrCreate()
    write_single_parquet(spark.createDataFrame(summary_data), output_path)
