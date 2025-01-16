from typing import List, Optional

import pyspark.sql as ps
from kedro.pipeline import Pipeline, node
from pyspark.sql import functions as F


def swap_columns_names(
    df: ps.DataFrame,
    source_col: str,
    target_col: str,
) -> ps.DataFrame:
    """Transformation to swap column names of PySpark Dataframe.

    Args:
        df: df to apply transformation on
        source_col: name of source col
        target_col: name of target col
    Returns:
        Dataframe where values of source_col and target_col are swapped
    """
    return (
        df.withColumnRenamed(source_col, "_tmp")
        .withColumnRenamed(target_col, source_col)
        .withColumnRenamed("_tmp", target_col)
    )


def join_and_collect_attributes(
    edges: ps.DataFrame, nodes: ps.DataFrame, col: str, attributes: List[str]
) -> ps.DataFrame:
    return (
        edges.alias("edges")
        .join(nodes.withColumn(col, F.col("id")), on=col, how="left")
        .withColumn(f"{col}_attributes", F.map_concat([F.create_map(F.lit(attr), F.col(attr)) for attr in attributes]))
        .select("edges.*", f"{col}_attributes")
    )


def generate_paths(
    pairs: ps.DataFrame, nodes: ps.DataFrame, edges: ps.DataFrame, attributes: Optional[List[str]] = None
) -> ps.DataFrame:
    attributes = ["id", "category"]

    # Construct hops dataframe
    paths = (
        edges.withColumn("is_forward", F.lit(True))
        .unionByName(edges.transform(swap_columns_names, "subject", "object").withColumn("is_forward", F.lit(False)))
        .transform(join_and_collect_attributes, nodes, "subject", attributes)
        .transform(join_and_collect_attributes, nodes, "object", attributes)
        .withColumn("path", F.array(F.col("subject_attributes"), F.col("object_attributes")))
        .select("subject", "object", "predicate", "path", "is_forward")
        .groupBy("subject", "object")
        .agg(
            # TODO: What do we do with the predicates though? Should this become list of lists in next iteration?
            F.collect_list(F.col("predicate")).alias("predicates"),
            F.collect_list(F.col("is_forward")).alias("is_forward"),
            F.first(F.col("path")).alias("path"),
        )
        # TODO: This is quick statement, that after path expansion allows us to detect cycles
        .withColumn("is_cycle", F.array_contains(F.transform(F.col("path"), lambda x: x["id"]), F.col("object")))
    )

    breakpoint()


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_paths,
                inputs={
                    "pairs": "integration.int.ground_truth.edges.norm@spark",
                    "nodes": "integration.int.rtx_kg2.nodes.norm@spark",
                    "edges": "integration.int.rtx_kg2.edges.norm@spark",
                },
                outputs=None,
            )
        ]
    )
