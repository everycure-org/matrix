import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class EmbiologyTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Embiology nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        EMBIOLOGY_SEPARATOR = r"\|"
        return (
            nodes_df
            .withColumn("upstream_data_source",              f.array(f.lit("embiology")))
            .withColumn("labels",                            f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("all_categories",                    f.split(f.col("all_categories") , EMBIOLOGY_SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_identifiers"), EMBIOLOGY_SEPARATOR))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.lit(None).cast(T.StringType())) # TODO: add international resource identifier
            .withColumn("description",                       f.lit(None).cast(T.StringType()))
        )
        # fmt: on

    def transform_edges(
        self,
        edges_df: ps.DataFrame,
        **kwargs,
    ) -> ps.DataFrame:
        """Transform Embiology edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
            pubmed_mapping: pubmed mapping
        Returns:
            Transformed DataFrame.
        """

        # fmt: off
        EMBIOLOGY_SEPARATOR = r"\|"
        return (
            edges_df
            # Qualifiers
            .withColumn("qualified_predicate",              f.lit(None).cast(T.StringType()))
            .withColumn("subject_aspect_qualifier",         f.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",      f.lit(None).cast(T.StringType()))
            .withColumn("subject_part_qualifier",           f.lit(None).cast(T.StringType()))
            .withColumn("object_aspect_qualifier",          f.lit(None).cast(T.StringType()))
            .withColumn("object_direction_qualifier",       f.lit(None).cast(T.StringType()))
            .withColumn("object_specialization_qualifier",  f.lit(None).cast(T.StringType()))
            .withColumn("object_part_qualifier",            f.lit(None).cast(T.StringType()))
            .withColumn("species_context_qualifier",        f.lit(None).cast(T.StringType()))
            .withColumn("disease_context_qualifier",        f.lit(None).cast(T.StringType()))
            .withColumn("frequency_qualifier",              f.lit(None).cast(T.StringType()))
            .withColumn("qualifiers",                       f.lit(None).cast(T.StringType()))
            .withColumn("stage_qualifier",                  f.lit(None).cast(T.StringType()))
            .withColumn("anatomical_context_qualifier",     f.lit(None).cast(T.StringType()))
            .withColumn("onset_qualifier",                  f.lit(None).cast(T.StringType()))
            .withColumn("sex_qualifier",                    f.lit(None).cast(T.StringType()))
            # Provenance
            .withColumn("knowledge_level",                  f.lit(None).cast(T.StringType()))
            .withColumn("agent_type",                       f.lit(None).cast(T.StringType()))
            .withColumn("primary_knowledge_source",         f.lit(None).cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("publications",                     f.split(f.col("publications"), EMBIOLOGY_SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("upstream_data_source",             f.array(f.lit("embiology")))
            # Quantitative attributes
            .withColumn("num_references",                   f.cast(T.IntegerType(), f.col("num_references")))
            .withColumn("num_sentences",                    f.cast(T.IntegerType(), f.col("num_sentences")))
            .withColumn("has_confidence_score",             f.lit(None).cast(T.FloatType()))
            .withColumn("extraction_confidence_score",      f.lit(None).cast(T.FloatType()))
            .withColumn("affinity",                         f.lit(None).cast(T.FloatType()))
            .withColumn("affinity_parameter",               f.lit(None).cast(T.StringType()))
            .withColumn("supporting_study_method_type",     f.lit(None).cast(T.StringType()))
        )
        # fmt: on
