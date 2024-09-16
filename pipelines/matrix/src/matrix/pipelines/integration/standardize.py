"""This module contains functions to standardize the input data to our target schema.

It uses pandera to validate the output and the pyspark.sql.functions to transform the data.
"""

import pandera.pyspark as pa
import pyspark.sql.types as T
from pandera.pyspark import DataFrameModel
from pandera.typing import Series
from typing import List
from matrix.schemas.data_api import KGEdgeSchema, KGNodeSchema
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
        .withColumn("upstream_kg_source", f.array(f.lit("robokop")))
        .withColumn("publications", f.split(f.col("publications"), "\x1f"))
        .withColumn("aggregator_knowledge_source", f.split(f.col("aggregator_knowledge_source"), "\x1f"))
        .withColumn("subject_aspect_qualifier", f.lit(None))  # FUTURE: not present in Robokop
        .withColumn("subject_direction_qualifier", f.lit(None))  # FUTURE: not present in Robokop
        # final selection of columns. 
        .select(
            "subject",
            "predicate",
            "object",
            "knowledge_level",
            "primary_knowledge_source",
            "aggregator_knowledge_source",
            "publications",
            "object_aspect_qualifier",
            "object_direction_qualifier",
            "subject_aspect_qualifier",
            "subject_direction_qualifier",
            "upstream_kg_source",
        )
    )
    # fmt: off


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
        nodes_df.withColumn("upstream_kg_source", f.array(f.lit("robokop")))
        .withColumn("all_categories", f.split(f.col("category"), "\x1f"))
        .withColumn( "equivalent_identifiers", f.split(f.col("equivalent_identifiers"), "\x1f"))
        .drop("category")
        .withColumn("category", f.col("all_categories").getItem(0))
        .withColumn("label", f.col("category"))
        .withColumn("publications", f.array(f.lit(None)))
        .withColumn("international_resource_identifier", f.lit(None))
        .select(
            "id",
            "name",
            "category",
            "description",
            "equivalent_identifiers",
            "all_categories",
            "publications",
            "label",
            "international_resource_identifier",
            "upstream_kg_source",
        )
    )
    # fmt: on
