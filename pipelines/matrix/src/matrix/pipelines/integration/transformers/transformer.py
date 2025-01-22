from abc import ABC, abstractmethod

import pyspark.sql as ps


class GraphTransformer(ABC):
    @abstractmethod
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """
        Function to transform nodes into the common format.

        Args:
            nodes_df: dataframe with nodes
        Returns:
            Nodes in standarized format
        """
        ...

    @abstractmethod
    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """
        Function to transform edges into the common format.

        Args:
            edges_df: dataframe with edges
        Returns:
            Edges in standarized format
        """
        ...
