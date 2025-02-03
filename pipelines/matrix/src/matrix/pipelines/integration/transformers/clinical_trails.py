import logging

import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class ClinicalTrialsTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
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
        return df
        # fmt: on

    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        """Transform edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
            pubmed_mapping: pubmed mapping
        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        df = (
            edges_df
            .withColumn("subject", f.col("drug_curie"))
            .withColumn("object", f.col("disease_curie"))
            # NOTE: Setting predicate such that it is unique
            .withColumn("predicate", f.lit("clinical_trails"))
            .filter((f.col("subject").isNotNull()) & (f.col("object").isNotNull()))
            .withColumn("significantly_better", f.col('significantly_better').cast('int'))
            .withColumn("significantly_worse", f.col('significantly_worse').cast('int'))
            .withColumn("non_significantly_worse", f.col('non_significantly_worse').cast('int'))
            .withColumn("non_significantly_better", f.col('non_significantly_better').cast('int'))
        )
        return df
