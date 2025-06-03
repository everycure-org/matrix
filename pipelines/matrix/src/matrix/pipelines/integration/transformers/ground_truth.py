import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class KGMLGroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def __init__(self, version: str):
        super().__init__()
        self._version = version

    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self._extract_edges(edges_df)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    @staticmethod
    def _extract_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("source", "subject")
            .withColumnRenamed("target", "object")
            .withColumn("disease", f.col("object"))
            .withColumn("drug", f.col("subject"))
            .withColumn(
                "y", f.when(f.col("indication").cast("boolean"), 1).when(f.col("contraindication").cast("boolean"), 0)
            )
            .withColumn("predicate", f.lit("clinical_trails"))
        )
