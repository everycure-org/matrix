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
            .distinct()
            .groupBy("curie") # Removes duplicates in the id column
            .agg(
                f.collect_list("name").alias("all_names"),
                f.first("name").alias("name")
            )
            .withColumnRenamed("curie", "id")
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
            .distinct()
            .withColumnRenamed("drug_curie", "subject")
            .withColumnRenamed("disease_curie", "object")
            .groupBy( # Removes duplicates in the subject and object columns
                # We do not aggregate outcome columns purposely, to not lose information
                "subject",
                "object",
                "significantly_better",
                "significantly_worse",
                "non_significantly_worse",
                "non_significantly_better",
            )
            .agg(
                f.collect_list("drug_name").alias("all_drug_names"),
                f.first("drug_name").alias("drug_name"),
                f.collect_list("disease_name").alias("all_disease_names"),
                f.first("disease_name").alias("disease_name"),
                f.collect_list("clinical_trial_id").alias("all_clinical_trial_ids"),
                f.first("clinical_trial_id").alias("clinical_trial_id"),
            )

            # FUTURE: Consider setting predicate based on outcome (significantly_better, significantly_worse, etc.)
            .withColumn("predicate", f.lit("clinical_trials"))
            .filter((f.col("subject").isNotNull()) & (f.col("object").isNotNull()))
            .withColumn("significantly_better", f.col('significantly_better').cast('int'))
            .withColumn("significantly_worse", f.col('significantly_worse').cast('int'))
            .withColumn("non_significantly_worse", f.col('non_significantly_worse').cast('int'))
            .withColumn("non_significantly_better", f.col('non_significantly_better').cast('int'))
        )
        return df
