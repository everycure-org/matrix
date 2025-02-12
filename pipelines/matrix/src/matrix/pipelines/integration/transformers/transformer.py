from abc import ABC, abstractmethod
from typing import Dict

import pyspark.sql as ps


class Transformer(ABC):
    """Data source transformer."""

    @abstractmethod
    def transform(self, **kwargs) -> Dict[str, ps.DataFrame]: ...


class GraphTransformer(Transformer):
    """Transformer for graph input sources."""

    def transform(self, nodes_df: ps.DataFrame, edges_df: ps.DataFrame, **kwargs) -> Dict[str, ps.DataFrame]:
        return {"nodes": self.transform_nodes(nodes_df, **kwargs), "edges": self.transform_edges(edges_df, **kwargs)}

    @abstractmethod
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame: ...

    @abstractmethod
    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame: ...
