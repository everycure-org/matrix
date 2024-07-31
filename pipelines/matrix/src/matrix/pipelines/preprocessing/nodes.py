"""Nodes for the preprocessing pipeline."""
import requests

import pandas as pd

from functools import partial


def resolve_curie(name: str, endpoint: str) -> str:
    """Function to retrieve curie through the synonymizer.

    FUTURE: Ensure downstream API yields 404 HTTP when not found.

    Args:
        name: name of the node
        endpoint: endpoint of the synonymizer
    Returns:
        Corresponding curie
    """
    result = requests.get(f"{endpoint}/synonymize", json={"name": name})

    element = result.json().get(name)
    if element:
        return element.get("preferred_curie", None)

    return None


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
