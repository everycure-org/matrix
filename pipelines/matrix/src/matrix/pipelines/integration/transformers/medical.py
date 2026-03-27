import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class MedicalTransformer(GraphTransformer):
    """Transformer for medical data."""

    def __init__(self, version: str, select_cols: str = True, drop_duplicates: bool = True):
        super().__init__(version, select_cols)
        self._drop_duplicates = drop_duplicates

    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        # fmt: off
        df = (
            nodes_df
            .withColumn("id",                                f.col("normalized_curie"))
            .withColumn("name",                              f.col("label"))
            .withColumn("upstream_data_source",              f.array(f.lit("ec_medical")))
            .withColumn("category",                          f.lit("category")) # FUTURE: Let's get rid of the category
            .withColumn("labels",                            f.array(f.col("types")))
            .withColumn("all_categories",                    f.array(f.col("types")))
            .withColumn("equivalent_identifiers",            f.array(f.col("id")))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.col("id"))
            # .transform(determine_most_specific_category, biolink_categories_df) need this?
            # Filter nodes we could not correctly resolve
            .filter(f.col("id").isNotNull())
        )
        # fmt: on

        if self._drop_duplicates:
            df = df.dropDuplicates(["id"])  # Drop any duplicate nodes

        return df

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        # fmt: off
        df = (
            edges_df
            # Core
            .withColumn("subject",                          f.col("SourceId"))
            .withColumn("object",                           f.col("TargetId"))
            .withColumn("predicate",                        f.concat(f.lit("biolink:"), f.lit(":"), f.col("Label")))
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
            .withColumn("primary_knowledge_source",         f.lit('medical team').cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",      f.array(f.lit('medical team')))
            .withColumn("publications",                     f.array(f.lit('medical team')))
            .withColumn("upstream_data_source",             f.array(f.lit("ec_medical")))
            # Quantitative attributes
            .withColumn("num_references",                   f.lit(None).cast(T.IntegerType()))
            .withColumn("num_sentences",                    f.lit(None).cast(T.IntegerType()))
            .withColumn("has_confidence_score",             f.lit(None).cast(T.FloatType()))
            .withColumn("extraction_confidence_score",      f.lit(None).cast(T.FloatType()))
            .withColumn("affinity",                         f.lit(None).cast(T.FloatType()))
            .withColumn("affinity_parameter",               f.lit(None).cast(T.StringType()))
            .withColumn("supporting_study_method_type",     f.lit(None).cast(T.StringType()))
            # Filter edges we could not correctly resolve
            .filter(f.col("subject").isNotNull() & f.col("object").isNotNull())
        )
        # fmt: on

        if self._drop_duplicates:
            df = df.dropDuplicates(["subject", "object", "predicate"])

        return df
