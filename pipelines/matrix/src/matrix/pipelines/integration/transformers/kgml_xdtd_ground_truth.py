import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class KGMLGroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def transform(self, positive_edges_df: DataFrame, negative_edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        pos_edges = self._extract_pos_edges(positive_edges_df)
        neg_edges = self._extract_neg_edges(negative_edges_df)
        edges = pos_edges.union(neg_edges)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges.withColumn("upstream_data_source", f.lit("kgml_xdtd"))}

    @staticmethod
    def _rename_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("drug_id", "subject")
            .withColumnRenamed("drug_ec_id", "subject_ec_id")
            .withColumnRenamed("disease_id", "object")
            .withColumn("predicate", f.lit("clinical_trials"))  # Is this correct? Should it be KGML? Ask Piotr
            .withColumn(
                "drug|disease", f.concat(f.col("subject"), f.lit("|"), f.col("object"))
            )  # Not sure where this column is used. Ask Piotr
        )

    def _extract_pos_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        edges_df = self._rename_edges(edges_df)
        return (
            edges_df.withColumn("indication", f.lit(True))
            .withColumn("contraindication", f.lit(False))
            .withColumn("y", f.lit(1))
        )

    def _extract_neg_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        edges_df = self._rename_edges(edges_df)
        return (
            edges_df.withColumn("indication", f.lit(False))
            .withColumn("contraindication", f.lit(True))
            .withColumn("y", f.lit(0))
        )
