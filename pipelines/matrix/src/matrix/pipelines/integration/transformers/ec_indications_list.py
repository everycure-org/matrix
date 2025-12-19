import logging

import pyspark.sql as ps
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class ECIndicationsListTransformer(Transformer):
    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self._extract_edges(edges_df)
        nodes = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _extract_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return edges_df.withColumnsRenamed(
            {"translator_id": "subject", "ec_id": "subject_ec_id", "target": "object"}
        ).withColumn("predicate", F.lit("JaM"))
