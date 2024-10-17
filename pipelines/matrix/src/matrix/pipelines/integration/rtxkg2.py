"""transformation functions for rtxkg2 nodes and edges."""

import pandera.pyspark as pa
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema


@pa.check_output(KGNodeSchema)
def transform_rtxkg2_nodes(nodes_df: DataFrame) -> DataFrame:
    """Transform RTX KG2 nodes to our target schema.

    Args:
        nodes_df: Nodes DataFrame.

    Returns:
        Transformed DataFrame.
    """
    SEP = "\u01c2"

    # fmt: off
    return (
        nodes_df
        .withColumn("upstream_data_source",              f.array(f.lit("rtxkg2")))
        .withColumn("labels",                            f.split(f.col("label"), SEP))
        .withColumn("all_categories",                    f.split(f.col("all_categories"), SEP))
        .withColumn("all_categories",                    f.array_distinct(f.concat("labels", "all_categories")))
        .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_curies"), SEP))
        .withColumn("publications",                      f.split(f.col("publications"), SEP))
        .withColumn("international_resource_identifier", f.col("iri"))
        # .withColumn("name", f.split(f.col("name"), "\x1f").getItem(0))
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


@pa.check_output(KGEdgeSchema)
def transform_rtxkg2_edges(edges_df: DataFrame) -> DataFrame:
    """Transform RTX KG2 edges to our target schema.

    Args:
        edges_df: Edges DataFrame.

    Returns:
        Transformed DataFrame.
    """
    SEP = "\u01c2"

    # fmt: off
    return (
        edges_df
        .withColumn("upstream_data_source",          f.array(f.lit("rtxkg2")))
        .withColumn("subject",                     f.col("subject"))
        .withColumn("object",                      f.col("object"))
        .withColumn("predicate",                   f.col("predicate"))
        .withColumn("knowledge_level",             f.lit(None))  # TODO / Not present in RTX KG2
        .withColumn("primary_knowledge_source",    f.col("knowledge_source"))
        .withColumn("aggregator_knowledge_source", f.array())
        .withColumn("publications",                f.split(f.col("publications"), SEP))
        .withColumn("subject_aspect_qualifier",    f.lit(None)) #not present in RTX KG2 v2.7, present in v2.10
        .withColumn("subject_direction_qualifier", f.lit(None)) #not present in RTX KG2 v2.7, present in v2.10
        .withColumn("object_aspect_qualifier",     f.lit(None)) #not present in RTX KG2 v2.7, present in v2.10
        .withColumn("object_direction_qualifier",  f.lit(None)) #not present in RTX KG2 v2.7, present in v2.10
        .select(*cols_for_schema(KGEdgeSchema))
    )
    # fmt: on
