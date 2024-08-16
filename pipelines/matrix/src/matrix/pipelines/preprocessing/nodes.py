"""Nodes for the preprocessing pipeline."""
import requests

import pandas as pd
import numpy as np

from typing import Callable, List, Optional
from functools import partial

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
    df: pd.DataFrame,
    endpoint: str,
    func: Callable,
    input_cols: str,
    target_col: str,
    coalesce_col: Optional[str] = None,
) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
        func: func to call
        input_cols: input cols, cols are coalesced to obtain single column
        target_col: target col
        coalesce_col: col to coalesce with if None
    Returns:
        dataframe enriched with Curie column
    """
    # Replace empty strings with nan
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Coalesce input cols
    col = coalesce(*[df[col] for col in input_cols])

    # Apply enrich function and replace nans by empty space
    df[target_col] = col.apply(partial(func, endpoint=endpoint))

    # If set, coalesce final result with coalesce col
    if coalesce_col:
        df[target_col] = coalesce(df[coalesce_col], df[target_col])

    # Ensure to fill nans with empty strings to avoid nans in nodebook
    return df.fillna("")


def create_prm_nodes(int_nodes: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary nodes dataset by filtering and renaming columns."""
    return (
        int_nodes[int_nodes["normalized_curie"].notna()]
        .drop(columns="curie")
        .rename(columns={"normalized_curie": "curie"})
    )


def create_int_edges(prm_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a intermediate edges dataset by combining primary nodes with intermediate edges."""
    index = prm_nodes[["ID", "curie"]]

    res = (
        int_edges.merge(
            index.rename(columns={"curie": "SourceId"}),
            left_on="Source",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
        .merge(
            index.rename(columns={"curie": "TargetId"}),
            left_on="Target",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
    )

    res["Included"] = res.apply(
        lambda row: not (pd.isna(row["SourceId"]) or pd.isna(row["TargetId"])), axis=1
    )

    return res.fillna("")


def create_prm_edges(int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary edges dataset by filtering and renaming columns."""
    # Replace empty strings with nan
    res = (
        int_edges.replace(r"^\s*$", np.nan, regex=True)
        .rename(
            columns={"SourceId": "subject", "TargetId": "object", "Label": "predicate"}
        )
        .dropna(subset=["subject", "object"])
    )

    res["predicate"] = "biolink:" + res["predicate"]
    res["knowledge_source"] = "EveryCure"

    return res
