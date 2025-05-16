import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class OffLabelTransformer(Transformer):
    """Transformer for off label data"""

    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self._extract_edges(edges_df)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    @staticmethod
    def _extract_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("curie_drug", "subject")
            .withColumnRenamed("curie_disease", "object")
            .withColumn("y", f.lit(1))  # all pairs are positive
            .withColumn("predicate", f.lit("off_label"))
        )
