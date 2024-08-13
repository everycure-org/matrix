"""Nodes for the preprocessing pipeline."""
import requests

import pandas as pd
import numpy as np

from typing import Callable, List
from functools import partial
from pyspark.sql import DataFrame

import pyspark.sql.functions as F

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key


def resolve(name: str, endpoint: str) -> str:
    """Function to retrieve curie through the synonymizer.

    Args:
        name: name of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    result = requests.get(f"{endpoint}/synonymize", json={"names": [name]})

    element = result.json().get(name)
    if element:
        return element.get("preferred_curie", None)

    return None


def normalize(curie: str, endpoint: str):
    """Function to retrieve the normalized identifier through the synonymizer.

    Args:
        curie: curie of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    if not curie or pd.isna(curie):
        return None

    result = requests.get(f"{endpoint}/normalize", json={"names": [curie]})

    element = result.json().get(curie)
    if element:
        return element.get("id", {}).get("identifier")

    return None


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


@has_schema(
    schema={
        "ID": "numeric",
        "name": "object",
        "curie": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["ID"])
def enrich_df(
    df: pd.DataFrame, endpoint: str, func: Callable, input_cols: str, target_col: str
) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
        func: func to call
        input_cols: input cols, cols are coalesced to obtain single column
        target_col: target col
    Returns:
        dataframe enriched with Curie column
    """
    # Replace empty strings with nan
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Coalesce input cols
    col = coalesce(*[df[col] for col in input_cols])

    # Apply enrich function and replace nans by empty space
    df[target_col] = col.apply(partial(func, endpoint=endpoint)).fillna("")

    return df


def create_prm_nodes(int_nodes: DataFrame) -> DataFrame:
    """Function to create prm nodes dataset.

    Args:
        int_nodes: int nodes dataset
    Returns:
        Primary nodes
    """
    return (
        int_nodes.filter(F.col("normalized_curie").isNotNull())
        .drop("curie")
        .withColumnRenamed("normalized_curie", "curie")
    )


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
