import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class OrchardTransformer(Transformer):
    """Transformer for orchard data"""

    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self._extract_edges(edges_df)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    @staticmethod
    def _extract_edges(edges_df: ps.DataFrame) -> ps.DataFrame:
        return (
            edges_df.withColumnRenamed("drug_kg_node_id", "subject")
            .withColumnRenamed("disease_kg_node_id", "object")
            .withColumn("predicate", f.lit("orchard"))
            .withColumn("upstream_data_source", f.array(f.lit("orchard")))
        )
