"""Modules with utilities for modelling."""
from functools import partial
import pandas as pd

from matrix.datasets.graph import KnowledgeGraph


def partial_(func: callable, **kwargs):
    """Function to wrap partial to enable partial function creation with kwargs.

    Args:
        func: Function to partially apply.
        kwargs: Keyword arguments to partially apply.

    Returns:
        Partially applied function.
    """
    return partial(func, **kwargs)


def _add_embeddings(data: pd.DataFrame, graph: KnowledgeGraph):
    """Adds columns containing knowledge graph embeddings.

    Args:
        data: Data to enrich with embeddings.
        graph: Knowledge graph.
    """
    data = data.copy()

    data["source_embedding"] = data.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    data["target_embedding"] = data.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )

    return data
