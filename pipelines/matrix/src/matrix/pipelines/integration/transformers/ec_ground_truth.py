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
        pos_edges = self._extract_positives(positive_edges)
        neg_edges = self._extract_negatives(negative_edges)
        edges = pos_edges.union(neg_edges).withColumn("upstream_source", f.lit(self.upstream_source))
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    def _extract_positives(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("final normalized drug id", "subject")
            .withColumnRenamed("final normalized disease id", "object")
            .withColumnRenamed("final normalized drug label", "subject_label")
            .withColumnRenamed("final normalized disease label", "object_label")
            .withColumnRenamed("drug|disease", "id")
            .withColumn("predicate", f.lit("indicated").cast(T.StringType()))
            .withColumn("y", f.lit(1))
            .withColumn("type", f.lit("indication"))
            .select("subject", "object", "subject_label", "object_label", "id", "predicate", "y", "type")
        )

    def _extract_negatives(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("final normalized drug id", "subject")
            .withColumnRenamed("final normalized disease id", "object")
            .withColumnRenamed("final normalized drug label", "subject_label")
            .withColumnRenamed("final normalized disease label", "object_label")
            .withColumnRenamed("drug|disease", "id")
            .withColumn("predicate", f.lit("contraindicated").cast(T.StringType()))
            .withColumn("y", f.lit(0))
            .withColumn("type", f.lit("contraindication"))
            .select("subject", "object", "subject_label", "object_label", "id", "predicate", "y", "type")
        )
