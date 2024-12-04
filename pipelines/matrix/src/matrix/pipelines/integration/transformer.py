from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class GraphTransformer(ABC):
    @abstractmethod
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        """
        Function to transform nodes into the common format.

        Args:
            nodes_df: dataframe with nodes
        Returns:
            Nodes in standarized format
        """
        ...

    @abstractmethod
    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        """
        Function to transform edges into the common format.

        Args:
            edges_df: dataframe with edges
        Returns:
            Edges in standarized format
        """
        ...
