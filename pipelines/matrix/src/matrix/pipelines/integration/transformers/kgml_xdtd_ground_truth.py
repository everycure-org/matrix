import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class GroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

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
        edges = pos_edges.union(neg_edges).withColumn("upstream_source", f.lit("kgml_xdtd"))
        id_list = edges.select("subject").union(edges.select("object")).withColumnRenamed("subject", "id").distinct()
        return {"nodes": id_list, "edges": edges}

    def _extract_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("source", "subject")
            .withColumnRenamed("target", "object")
            .withColumn("id", f.concat_ws("|", f.col("subject"), f.col("object")))
            .withColumn("subject_label", f.lit(None).cast(T.StringType()))
            .withColumn("object_label", f.lit(None).cast(T.StringType()))
            .withColumn("flag", f.lit(None).cast(T.StringType()))
            .select("subject", "object", "subject_label", "object_label", "id", "flag")
        )
