import logging

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class GroundTruthTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return nodes_df
        # fmt: on

    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        df = (
            edges_df.withColumn("disease", f.col("object"))
            .withColumn("drug", f.col("subject"))
            .withColumn(
                "y", f.when(f.col("indication").cast("boolean"), 1).when(f.col("contraindication").cast("boolean"), 0)
            )
            .withColumn("predicate", f.lit("clinical_trails"))
        )
        return df
