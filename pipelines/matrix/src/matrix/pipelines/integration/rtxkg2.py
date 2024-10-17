"""transformation functions for rtxkg2 nodes and edges."""

from typing import Any, Dict

import pandas as pd

import pandera.pyspark as pa
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .biolink import unnest
from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

RTX_SEPARATOR = "\u01c2"


@pa.check_output(KGNodeSchema)
def transform_rtxkg2_nodes(nodes_df: DataFrame) -> DataFrame:
    """Transform RTX KG2 nodes to our target schema.

    Args:
        nodes_df: Nodes DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # fmt: off
    return (
        nodes_df
        .withColumn("upstream_data_source",              f.array(f.lit("rtxkg2")))
        .withColumn("labels",                            f.split(f.col(":LABEL"), RTX_SEPARATOR))
        .withColumn("all_categories",                    f.split(f.col("all_categories:string[]"), RTX_SEPARATOR))
        .withColumn("all_categories",                    f.array_distinct(f.concat("labels", "all_categories")))
        .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_curies:string[]"), RTX_SEPARATOR))
        .withColumn("publications",                      f.split(f.col("publications:string[]"), RTX_SEPARATOR))
        .withColumn("international_resource_identifier", f.col("iri"))
        .withColumnRenamed("id:ID", "id")
        .withColumnRenamed("name", "name")
        .withColumnRenamed("category", "category")
        .withColumnRenamed("description", "description")
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


def create_biolink_hierarchy(biolink_predicates: Dict[str, Any]) -> pd.DataFrame:
    return unnest(biolink_predicates)


@pa.check_output(KGEdgeSchema)
def transform_rtxkg2_edges(edges_df: DataFrame, biolink_hierarchy: DataFrame) -> DataFrame:
    """Transform RTX KG2 edges to our target schema.

    Args:
        edges_df: Edges DataFrame.

    Returns:
        Transformed DataFrame.
    """

    # Filter hierarical edges
    edges_df = edges_df.withColumn("clean_predicate", f.regexp_replace("predicate", "^biolink:")).join(
        biolink_hierarchy
    )

    # fmt: off
    return (
        edges_df
        .withColumn("upstream_data_source",          f.array(f.lit("rtxkg2")))
        .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
        .withColumn("primary_knowledge_source",      f.element_at(f.split(f.col("knowledge_source:string[]"), RTX_SEPARATOR), 1))
        .withColumn("aggregator_knowledge_source",   f.slice(f.split(f.col("knowledge_source:string[]"), RTX_SEPARATOR), 2, 100))
        .withColumn("publications",                  f.split(f.col("publications:string[]"), RTX_SEPARATOR))
        .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType())) #not present in RTX KG2 at this time
        .select(*cols_for_schema(KGEdgeSchema))
    )
    # fmt: on
