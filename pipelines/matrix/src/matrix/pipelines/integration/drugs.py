import logging
import pyspark.sql.functions as f
import pandera.pyspark as pa
from pyspark.sql import DataFrame

from .transformer import GraphTransformer
from matrix.schemas.knowledge_graph import NodeSchema

logger = logging.getLogger(__name__)


class DrugsTransformer(GraphTransformer):
    @pa.check_output(NodeSchema)
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
            .withColumn("id",                                f.col("curie"))
            .withColumn("name",                              f.col("curie_label"))
            .withColumn("category",                          f.lit("biolink:Drug"))
        )
        return df
        # fmt: on

    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        raise NotImplementedError("Not implemented!")
