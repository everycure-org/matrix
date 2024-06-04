"""Nodes for the ingration pipeline."""
from typing import List
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StringType, StructField

from refit.v1.core.inline_has_schema import has_schema
from matrix.datasets.neo4j import cypher_query


@has_schema(
    schema={
        "id": "string",
    },
    allow_subset=True,
    relax=True,
)
def extract_nodes(raw_nodes: DataFrame, types: List[str]) -> DataFrame:
    """Function to extract nodes with given type.

    Args:
        raw_nodes: Raw nodes DataFrame.
        types: List of types to filter.
    """
    return raw_nodes.filter(F.col("category").isin(types))


@has_schema(
    schema={
        "source_id": "string",
        "target_id": "string",
    },
    allow_subset=True,
    relax=True,
)
def extract_edges(raw_edges: DataFrame) -> DataFrame:
    """Function to extract edges.

    Args:
        raw_edges: Raw edges DataFrame.
    """
    return raw_edges


def apply_date_filter(data: DataFrame, cutoff_date: datetime) -> None:
    """Function to create int embeddings.

    Args:
        data: Dataframe
        cutoff_date: Cutoff date
    """
    return data.filter(F.col("date_discovered") <= cutoff_date)


@cypher_query(
    query="""
        MATCH (drug:`Drug`)-[:`TREATS`]->(disease:`Disease`) 
        RETURN 
            drug.`id` AS drug_label, 
            disease.`id` AS disease_label, 
            disease.`date_discovered` AS datetime_str
    """,
    schema=StructType(
        [
            StructField("drug_label", StringType(), True),
            StructField("disease_label", StringType(), True),
            StructField("datetime_str", StringType(), True),
        ]
    ),
)
def neo4j_decorated(data: DataFrame, drug_types: List[str]):
    """Function to retrieve Neo4J data.

    Args:
        data: Dataframe representing query result
        drug_types: additional arg
    """
    print(drug_types)
    data.show()
