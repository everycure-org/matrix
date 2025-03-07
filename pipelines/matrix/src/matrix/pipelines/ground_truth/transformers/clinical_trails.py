import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import Transformer

logger = logging.getLogger(__name__)


class ClinicalTrialsTransformer(Transformer):
    def __init__(self, drop_duplicates: bool = True):
        super().__init__()
        self._drop_duplicates = drop_duplicates

    def transform(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        # fmt: off
        df = (
            edges_df
            .withColumn("subject", f.col("drug_curie"))
            .withColumn("object", f.col("disease_curie"))
            .withColumn("subject_label", f.col("drug_name"))
            .withColumn("object_label", f.col("disease_name"))
            .withColumn("predicate", f.lit("clinical_trails"))
            .withColumn("id", f.concat_ws("|", f.col("subject"), f.col("object")))
            .withColumn("y", f.when(f.col("significantly_better").cast("boolean"), 1).when(f.col("significantly_worse").cast("boolean"), 0).otherwise(None))
            .withColumn("upstream_source", f.lit("ec_clinical_trial"))
            .withColumn("type",f.when(f.col("significantly_better")==1,'significantly_better')
                .when(f.col("significantly_worse")==1,'significantly_worse')
                .when(f.col("non_significantly_worse")==1,'non_significantly_worse')
                .when(f.col("non_significantly_better")==1,'non_significantly_better')
                .otherwise(None))
            .select("subject", "object", "subject_label", "object_label", "id", "predicate", "y", "type","upstream_source")
        )

        if self._drop_duplicates:
            df = df.dropDuplicates(["subject", "object", "predicate", "type"])
        id_list = df.select("subject").union(df.select("object")).withColumnRenamed("subject", "id").distinct()
        return {"nodes": id_list, "edges": df}
        # fmt: on
