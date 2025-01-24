import logging

import pandera.pyspark as pa
import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class MedicalTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        df = (
            nodes_df
            .withColumnRenamed("normalized_curie", "id")
            .distinct()
            .groupBy("id") # Removes duplicates in the id column
            .agg(
                f.collect_list("label").alias("all_names"),
                f.first("label").alias("name"),
                f.collect_list("types").alias("labels"),
            )
            .withColumn("upstream_data_source",              f.array(f.lit("ec_medical")))
            .withColumn("category",                          f.lit("category")) # FUTURE: Let's get rid of the category
            .withColumn("all_categories",                    f.col("labels"))
            .withColumn("equivalent_identifiers",            f.array(f.col("id")))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.col("id"))
            # .transform(determine_most_specific_category, biolink_categories_df) need this?
            # Filter nodes we could not correctly resolve
            .filter(f.col("id").isNotNull())
        )
        return df
        # fmt: on

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
            pubmed_mapping: pubmed mapping
        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        edges = (
            edges_df
            .withColumnRenamed("SourceId", "subject")
            .withColumnRenamed("TargetId", "object")
            .withColumn("predicate",                     f.concat(f.lit("biolink:"), f.lit(":"), f.col("Label")))
            .distinct()
            .groupBy("subject", "object") # Removes duplicates in the subject and object columns
            .agg(
                f.collect_list("predicate").alias("all_predicates"),
                f.first("predicate").alias("predicate"),
            )
            .withColumn("upstream_data_source",          f.array(f.lit("ec_medical")))
            .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",   f.array(f.lit('medical team')))
            .withColumn("primary_knowledge_source",      f.lit('medical team').cast(T.StringType()))
            .withColumn("publications",                  f.array(f.lit('medical team')))
            .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType())) #not present
            .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType())) #not present
            .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType())) #not present
            .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType())) #not present
            
            # Filter edges we could not correctly resolve
            .filter(f.col("subject").isNotNull() & f.col("object").isNotNull() & f.col("predicate").isNotNull())
        )
        return edges
