"""Nodes for the preprocessing pipeline."""
import requests

import pandas as pd

from functools import partial
from pyspark.sql import DataFrame

import pyspark.sql.functions as F

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key


def resolve_curie(name: str, endpoint: str) -> str:
    """Function to retrieve curie through the synonymizer.

    FUTURE: Ensure downstream API yields 404 HTTP when not found.

    Args:
        name: name of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    # For instance, I give {"gives": ["long covid"]}
    result = requests.get(f"{endpoint}/synonymize", json={"names": [name]})

    # Retrsult {"long covid": {"preferred_curie": "MONDO:x"}}
    element = result.json().get(name)
    if element:
        return element.get("preferred_curie", None)

    return None


@has_schema(
    schema={
        "ID": "numeric",
        "name": "object",
        "curie": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["ID"])
def resolve_nodes(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
    Returns:
        dataframe enriched with Curie column
    """
    df["curie"] = df["name"].apply(partial(resolve_curie, endpoint=endpoint))

    return df


def create_prm_nodes(int_nodes: DataFrame) -> DataFrame:
    """Function to create prm nodes dataset.

    Args:
        int_nodes: int nodes dataset
    Returns:
        Primary nodes
    """
    return int_nodes.filter(F.col("curie").isNotNull())


def create_prm_edges(prm_nodes: DataFrame, int_edges: DataFrame) -> DataFrame:
    """Function to create prm edges dataset.

    Args:
        prm_nodes: primary nodes dataset
        int_edges: int edges dataset
    Returns:
        Primary nodes
    """
    index = prm_nodes.select("ID", "curie")

    res = (
        int_edges.join(
            index.withColumnRenamed("curie", "subject"),
            int_edges.Source == prm_nodes.ID,
            how="left",
        )
        .drop("ID")
        .join(
            index.withColumnRenamed("curie", "object"),
            int_edges.Target == prm_nodes.ID,
            how="left",
        )
        .drop("ID")
        .filter(F.col("subject").isNotNull() & F.col("object").isNotNull())
        .withColumn("predicate", F.concat(F.lit("biolink:"), F.col("Label")))
        .withColumn("knowledge_source", F.lit("EveryCure"))
    )

    return res
