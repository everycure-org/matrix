import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class ECGroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def transform(self, positive_edges_df: DataFrame, negative_edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        pos_edges = self._extract_pos_edges(positive_edges_df)
        neg_edges = self._extract_neg_edges(negative_edges_df)
        edges = pos_edges.union(neg_edges)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges.withColumn("upstream_data_source", f.lit("ec"))}

    @staticmethod
    def _rename_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("final normalized drug id", "subject")
            .withColumnRenamed("drug_ec_id", "subject_ec_id")
            .withColumnRenamed("final normalized disease id", "object")
            .withColumnRenamed("final normalized drug label", "subject_label")
            .withColumnRenamed("final normalized disease label", "object_label")
        )

    def _extract_pos_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        edges_df = self._rename_edges(edges_df)
        return (
            edges_df.select("subject", "subject_ec_id", "object", "subject_label", "object_label")
            .withColumn("predicate", f.lit("indication"))
            .withColumn("y", f.lit(1))
        )

    def _extract_neg_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        edges_df = self._rename_edges(edges_df)
        return (
            edges_df.select("subject", "subject_ec_id", "object", "subject_label", "object_label")
            .withColumn("predicate", f.lit("contraindication"))
            .withColumn("y", f.lit(0))
        )
