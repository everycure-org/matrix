import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class OrchardTransformer(Transformer):
    """Transformer for off label data"""

    def __init__(self, version: str, pair_flags: dict[str, str]):
        super().__init__(version)
        self._pair_flags = pair_flags

    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self._extract_edges(self, edges_df)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    @staticmethod
    def _extract_edges(self, edges_df: ps.DataFrame) -> ps.DataFrame:
        edges_df = (
            edges_df.withColumnRenamed("drug_kg_node_id", "subject")
            .withColumnRenamed("disease_kg_node_id", "object")
            .withColumn("upstream_data_source", f.array(f.lit("orchard")))
            .withColumn("predicate", f.lit("orchard"))
        )
        # Apply dynamic filters based on pair_flags configuration in integration/parameters.yml
        for flag_name, filter_dict in self._pair_flags.items():
            # For each flag name, create a list of filter conditions using list comprehensiom from filter_dict object
            filter_conditions = [
                f.col(filter_col).isin(filter_values) for filter_col, filter_values in filter_dict.items()
            ]

            # Generate a master condition object for filtering by combining all filters with an AND operator
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition = combined_condition & condition

            # Apply the master condition to generate a flag_name column
            edges_df = edges_df.withColumn(flag_name, f.when(combined_condition, f.lit(1)).otherwise(f.lit(0)))
        columns_to_drop = set([col for flag in self._pair_flags.values() for col in flag.keys()])
        return edges_df.drop(*columns_to_drop)
