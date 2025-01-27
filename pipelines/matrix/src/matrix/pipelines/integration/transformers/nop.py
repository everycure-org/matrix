import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class NopTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """
        Function to transform nodes into the common format.

        Args:
            nodes_df: dataframe with nodes
        Returns:
            Nodes in standarized format
        """
        return nodes_df

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """
        Function to transform edges into the common format.

        Args:
            edges_df: dataframe with edges
        Returns:
            Edges in standarized format
        """
        return edges_df
