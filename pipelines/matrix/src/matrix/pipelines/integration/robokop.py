"""transformation functions for robokop nodes and edges."""

import pandera.pyspark as pa
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

# FUTURE: We should likely not need to rename these columns as we do below
# However, KGX is currently not as performant as we need it to be thus
# we do it manually with spark. This ought to be improved, e.g. by
# adding parquet support to KGX.
# or implementing a custom KGX version that leverages spark for higher performance
# https://github.com/everycure-org/matrix/issues/474


@pa.check_output(KGEdgeSchema)
def transform_robo_edges(edges_df: DataFrame) -> DataFrame:
    """Transform Robokop edges to our target schema.

    Args:
        edges_df: Edges DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # NOTE: This function was partially generated using AI assistance.
    # fmt: off
    return (
        edges_df
        .withColumnRenamed("subject:START_ID",                  "subject")
        .withColumnRenamed("predicate:TYPE",                    "predicate")
        .withColumnRenamed("object:END_ID",                     "object")
        .withColumnRenamed("primary_knowledge_source:string",   "primary_knowledge_source")
        .withColumnRenamed("knowledge_level:string",            "knowledge_level")
        .withColumnRenamed("object_aspect_qualifier:string",    "object_aspect_qualifier")
        .withColumnRenamed("object_direction_qualifier:string", "object_direction_qualifier")
        .withColumn("upstream_data_source",                      f.array(f.lit("robokop")))
        .withColumn("publications",                             f.split(f.col("publications:string[]"), "\x1f"))
        .withColumn("aggregator_knowledge_source",              f.split(f.col("aggregator_knowledge_source:string[]"), "\x1f"))
        .withColumn("subject_aspect_qualifier",                 f.lit(None).cast(T.StringType()))
        .withColumn("subject_direction_qualifier",              f.lit(None).cast(T.StringType()))
        # final selection of columns
        .select(*cols_for_schema(KGEdgeSchema))
    )
    # fmt: on


@pa.check_output(KGNodeSchema)
def transform_robo_nodes(nodes_df: DataFrame) -> DataFrame:
    """Transform Robokop nodes to our target schema.

    Args:
        nodes_df: Nodes DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # NOTE: This function was partially generated using AI assistance.
    # fmt: off
    return (
        nodes_df.withColumn("upstream_data_source",       f.array(f.lit("robokop")))
        .withColumn("all_categories",                    f.split(f.col("category:LABEL"), "\x1f"))
        .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_identifiers:string[]"), "\x1f"))
        .withColumn("category",                          f.element_at(f.col("all_categories"), f.size(f.col("all_categories"))))
        .withColumn("labels",                            f.array(f.col("all_categories")))
        .withColumn("publications",                      f.array(f.lit(None)))
        .withColumn("international_resource_identifier", f.lit(None))
        .withColumnRenamed("id:ID", "id")
        .withColumnRenamed("name:string", "name")
        .withColumnRenamed("description:string", "description")
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on
