from abc import ABC, abstractmethod
from typing import Dict, List

import pyspark.sql as ps

from matrix.pipelines.integration.schema import BIOLINK_KG_EDGE_SCHEMA, BIOLINK_KG_NODE_SCHEMA


class Transformer(ABC):
    """Data source transformer."""

    def __init__(self, version: str):
        super().__init__()
        self._version = version

    @abstractmethod
    def transform(self, **kwargs) -> Dict[str, ps.DataFrame]: ...


def select_if(df: ps.DataFrame, cols: List[str], cond: bool):
    if cond:
        return df.select(*cols)

    return df


class GraphTransformer(Transformer):
    """Transformer for graph input sources."""

    def __init__(self, version: str, select_cols: str = True) -> None:
        self._select_cols = select_cols
        super().__init__(version)

    def transform(self, nodes_df: ps.DataFrame, edges_df: ps.DataFrame, **kwargs) -> Dict[str, ps.DataFrame]:
        return {
            "nodes": self.transform_nodes(nodes_df, **kwargs).transform(
                select_if, cols=BIOLINK_KG_NODE_SCHEMA.columns.keys(), cond=self._select_cols
            ),
            "edges": self.transform_edges(edges_df, **kwargs).transform(
                select_if, cols=BIOLINK_KG_EDGE_SCHEMA.columns.keys(), cond=self._select_cols
            ),
        }

    @abstractmethod
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame: ...

    @abstractmethod
    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame: ...
