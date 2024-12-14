import logging
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class GroundTruthTransformer(GraphTransformer):
    # @pa.check_output(KGNodeSchema)
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        df = (
            nodes_df
            .withColumn("target", f.col("disease"))
            .withColumn("source", f.col("drug"))
            .withColumn("y",
                       f.when(f.col("indication").cast('boolean'), 1)
                       .when(f.col("contraindication").cast('boolean'), 0))
        )
        df.printSchema()
        return df
        # fmt: on

    # @pa.check_output(KGEdgeSchema)
    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        raise NotImplementedError("Not implemented!")
