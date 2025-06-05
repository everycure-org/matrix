import logging

import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class ClinicalTrialsTransformer(GraphTransformer):
    def __init__(self, select_cols: str = True, drop_duplicates: bool = True):
        super().__init__(select_cols)
        self._drop_duplicates = drop_duplicates

    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        # fmt: off
        df = (
            nodes_df
            .withColumn("id",                                f.col("curie"))
            .withColumn("upstream_data_source",              f.array(f.lit("ec_clinical_trails")))
            .withColumn("labels",                            f.array(f.lit("entity label"))) # TODO: Fix entity labels for medical?
            .withColumn("all_categories",                    f.array(f.lit("biolink:"))) # TODO fix
            .withColumn("equivalent_identifiers",            f.array(f.col("id")))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.col("id"))
            # .transform(determine_most_specific_category, biolink_categories_df) need this?
            # Filter nodes we could not correctly resolve
            .filter(f.col("id").isNotNull())
        )
        if self._drop_duplicates:
            df = df.dropDuplicates(["id"])  # Drop any duplicate nodes
        return df

    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        # fmt: off
        df = (
            edges_df
            .withColumn("subject", f.col("drug_curie"))
            .withColumn("object", f.col("disease_curie"))
            .withColumn("predicate", f.lit("clinical_trails"))
            .filter((f.col("subject").isNotNull()) & (f.col("object").isNotNull()))
            .withColumn("significantly_better", f.col('significantly_better').cast('int'))
            .withColumn("significantly_worse", f.col('significantly_worse').cast('int'))
            .withColumn("non_significantly_worse", f.col('non_significantly_worse').cast('int'))
            .withColumn("non_significantly_better", f.col('non_significantly_better').cast('int'))
        )

        if self._drop_duplicates:
            df = df.dropDuplicates(["subject", "object", "predicate"])

        return df
        # fmt: on
