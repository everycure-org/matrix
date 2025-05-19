import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class GroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def __init__(self, upstream_source: str = "matrix_indication_list", **kwargs):
        super().__init__(**kwargs)
        self.upstream_source = upstream_source

    def transform(self, positive_edges: DataFrame, negative_edges: DataFrame, **kwargs) -> dict[str, DataFrame]:
        pos_edges = (
            self._extract_edges(positive_edges)
            .withColumn("predicate", f.lit("indication").cast(T.StringType()))
            .withColumn("y", f.lit(1))
        )
        neg_edges = (
            self._extract_edges(negative_edges)
            .withColumn("predicate", f.lit("contraindication").cast(T.StringType()))
            .withColumn("y", f.lit(0))
        )
        edges = pos_edges.union(neg_edges).withColumn("upstream_source", f.lit(self.upstream_source))
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    def _extract_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("final normalized drug id", "subject")
            .withColumnRenamed("final normalized disease id", "object")
            .withColumnRenamed("final normalized drug label", "subject_label")
            .withColumnRenamed("final normalized disease label", "object_label")
            .withColumnRenamed("downfilled from mondo", "flag")
            .withColumnRenamed("drug|disease", "id")
            .filter(~edges_df["flag"])
            .select("subject", "object", "subject_label", "object_label", "id", "flag")
        )
