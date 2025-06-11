import logging
from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as F

from .transformer import Transformer

logger = logging.getLogger(__name__)


class DrugMechTransformer(Transformer):
    def transform(self, edges_df: ps.DataFrame, **kwargs) -> Dict[str, ps.DataFrame]:
        edges = self._extract_edges(edges_df)
        nodes = (
            edges.withColumn("id", F.col("subject"))
            .union(edges.withColumn("id", F.col("object")))
            .select("id")
            .distinct()
        )

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _extract_edges(edges: ps.DataFrame) -> ps.DataFrame:
        return (
            edges.withColumn("graph_id", F.col("graph").getItem("_id"))
            .select("graph_id", F.explode("links").alias("links"))
            .withColumn("subject", F.col("links").getItem("source"))
            .withColumn("object", F.col("links").getItem("target"))
            .withColumn("predicate", F.col("links").getItem("key"))
            .select("graph_id", "subject", "predicate", "object")
            .distinct()
        )
