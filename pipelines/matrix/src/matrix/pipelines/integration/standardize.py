"""This module contains functions to standardize the input data to our target schema.

It uses pandera to validate the output and the pyspark.sql.functions to transform the data.
"""

import pandera.pyspark as pa
import pyspark.sql.types as T
from pandera.pyspark import DataFrameModel
from pandera.typing import Series
from typing import List
from matrix.schemas.data_api import KGEdgeSchema, KGNodeSchema, cols_for_schema
from pyspark.sql import DataFrame
import pyspark.sql.functions as f


@pa.check_output(KGEdgeSchema)
def transform_robo_edges(edges_df: DataFrame) -> DataFrame:
    """Transform Robokop edges to our target schema.

    Args:
        edges_df: Edges DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # fmt: off
    return (
        edges_df
        .withColumn("upstream_kg_source",          f.array(f.lit("robokop")))
        .withColumn("publications",                f.split(f.col("publications"), "\x1f"))
        .withColumn("aggregator_knowledge_source", f.split(f.col("aggregator_knowledge_source"), "\x1f"))
        .withColumn("subject_aspect_qualifier",    f.lit(None))  # FUTURE: not present in Robokop
        .withColumn("subject_direction_qualifier", f.lit(None))  # FUTURE: not present in Robokop
        # final selection of columns. 
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
    # fmt: off
    return (
        nodes_df.withColumn("upstream_kg_source",        f.array(f.lit("robokop")))
        .withColumn("all_categories",                    f.split(f.col("category"), "\x1f"))
        .withColumn( "equivalent_identifiers",           f.split(f.col("equivalent_identifiers"), "\x1f"))
        .withColumn("category",                          f.col("all_categories").getItem(0))
        .withColumn("label",                             f.col("category"))
        .withColumn("publications",                      f.array(f.lit(None)))
        .withColumn("international_resource_identifier", f.lit(None))
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


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
        .withColumn("upstream_kg_source",                f.array(f.lit("rtxkg2")))
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
        .withColumn("upstream_kg_source",          f.array(f.lit("rtxkg2")))
        .withColumn("subject",                     f.col("subject"))
        .withColumn("object",                      f.col("object"))
        .withColumn("predicate",                   f.col("predicate"))
        .withColumn("knowledge_level",             f.lit(None))  # TODO / Not present in RTX KG2
        .withColumn("primary_knowledge_source",    f.col("knowledge_source"))
        .withColumn("aggregator_knowledge_source", f.array())
        .withColumn("publications",                f.split(f.col("publications"), SEP))
        .withColumn("subject_aspect_qualifier",    f.lit(None))
        .withColumn("subject_direction_qualifier", f.lit(None))
        .withColumn("object_aspect_qualifier",     f.lit(None))
        .withColumn("object_direction_qualifier",  f.lit(None))
        .select(*cols_for_schema(KGEdgeSchema))
    )
    # fmt: on
