from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class GraphTransformer(ABC):
    @abstractmethod
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame: ...

    @abstractmethod
    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame: ...
