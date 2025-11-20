import logging

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class ClinicalTrialsTransformer(Transformer):
    def __init__(self, version: str, drop_duplicates: bool = True):
        super().__init__(version)
        self._drop_duplicates = drop_duplicates

    def transform(self, edges_df: DataFrame, **kwargs) -> dict[str, DataFrame]:
        edges = self.transform_edges(edges_df, **kwargs)
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        # fmt: off
        df = (
            edges_df
            .withColumn("subject",                      f.col("drug_ec_id"))
            .withColumn("object",                       f.col("disease_curie"))
            .withColumn("predicate",                    f.lit("clinical_trials"))
            .filter((f.col("subject").isNotNull()) & (f.col("object").isNotNull()))
            .withColumn("upstream_data_source",         f.array(f.lit("clinical_trials")))
            .withColumn("significantly_better",         f.col('significantly_better').cast('int'))
            .withColumn("significantly_worse",          f.col('significantly_worse').cast('int'))
            .withColumn("non_significantly_worse",      f.col('non_significantly_worse').cast('int'))
            .withColumn("non_significantly_better",     f.col('non_significantly_better').cast('int'))
        )
        # fmt: on

        if self._drop_duplicates:
            df = df.dropDuplicates(["subject", "object", "predicate"])

        return df
