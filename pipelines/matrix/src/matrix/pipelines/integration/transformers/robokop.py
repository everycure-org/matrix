import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

# FUTURE: We should likely not need to rename these columns as we do below
# However, KGX is currently not as performant as we need it to be thus
# we do it manually with spark. This ought to be improved, e.g. by
# adding parquet support to KGX.
# or implementing a custom KGX version that leverages spark for higher performance
# https://github.com/everycure-org/matrix/issues/474
from matrix.pipelines.integration.filters import determine_most_specific_category

from .transformer import GraphTransformer

ROBOKOP_SEPARATOR = r"\|"


class RobokopTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "c5ec1f282158182f":
                df = transform_nodes_c5ec1f282158182f(nodes_df)
            case "30fd1bfc18cd5ccb.1":
                df = transform_nodes_30fd1bfc18cd5ccb_1(nodes_df)
            case "30fd1bfc18cd5ccb":
                df = transform_nodes_30fd1bfc18cd5ccb(nodes_df)
            case _:
                raise NotImplementedError(f"No nodes transformer code implemented for version: {self._version}")

        return df

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop edges to our target schema.

        Args:
            edges_df: Edges DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "c5ec1f282158182f":
                df = transform_edges_c5ec1f282158182f(edges_df)
            case "30fd1bfc18cd5ccb" | "30fd1bfc18cd5ccb.1":
                df = transform_edges_30fd1bfc18cd5ccb(edges_df)
            case _:
                raise NotImplementedError(f"No edges transformer code implemented for version: {self._version}")
        return df


def transform_nodes_c5ec1f282158182f(nodes_df: ps.DataFrame):
    # fmt: off
    df = (nodes_df
          .withColumn("upstream_data_source",              F.array(F.lit("robokop")))
          .withColumn("all_categories",                    F.split(F.col("category:LABEL"), ROBOKOP_SEPARATOR))
          .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers:string[]"), ROBOKOP_SEPARATOR))
          .withColumn("labels",                            F.col("all_categories"))
          .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
          .withColumnRenamed("id:ID", "id")
          .withColumnRenamed("name:string", "name")
          .withColumnRenamed("description:string", "description")
          # getting most specific category
          .transform(determine_most_specific_category))
    # fmt: on
    return df


def transform_nodes_30fd1bfc18cd5ccb_1(nodes_df: ps.DataFrame):
    # fmt: off
    df = (nodes_df
          .withColumn("upstream_data_source",              F.array(F.lit("robokop")))
          .withColumn("all_categories",                    F.split(F.col("category"), ROBOKOP_SEPARATOR))
          .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers"), ROBOKOP_SEPARATOR))
          .withColumn("labels",                            F.col("all_categories"))
          .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
          # getting most specific category
          .transform(determine_most_specific_category))
    # fmt: on
    return df


def transform_nodes_30fd1bfc18cd5ccb(nodes_df: ps.DataFrame):
    # fmt: off
    df = (nodes_df
          .withColumn("upstream_data_source",              F.array(F.lit("robokop")))
          .withColumn("all_categories",                    F.split(F.col("category"), ROBOKOP_SEPARATOR))
          .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers"), ROBOKOP_SEPARATOR))
          .withColumn("labels",                            F.col("all_categories"))
          .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
          # getting most specific category
          .transform(determine_most_specific_category)
    )
    # fmt: on
    return df


def transform_edges_30fd1bfc18cd5ccb(edges_df: ps.DataFrame):
    # fmt: off
    df = (edges_df
          # Qualifiers
          .withColumn("qualified_predicate",                      F.col("qualified_predicate").cast(T.StringType()))
          .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
          .withColumn("subject_part_qualifier",                   F.col("subject_part_qualifier").cast(T.StringType()))
          .withColumn("object_aspect_qualifier",                  F.col("object_aspect_qualifier").cast(T.StringType()))
          .withColumn("object_direction_qualifier",               F.col("object_direction_qualifier").cast(T.StringType()))
          .withColumn("object_specialization_qualifier",          F.col("object_specialization_qualifier").cast(T.StringType()))
          .withColumn("object_part_qualifier",                    F.col("object_part_qualifier").cast(T.StringType()))
          .withColumn("species_context_qualifier",                F.col("species_context_qualifier").cast(T.StringType()))
          .withColumn("disease_context_qualifier",                F.col("disease_context_qualifier").cast(T.StringType()))
          .withColumn("frequency_qualifier",                      F.col("frequency_qualifier").cast(T.StringType()))
          .withColumn("qualifiers",                               F.col("qualifiers").cast(T.StringType()))
          .withColumn("stage_qualifier",                          F.col("stage_qualifier").cast(T.StringType()))
          .withColumn("anatomical_context_qualifier",             F.col("anatomical_context_qualifier").cast(T.StringType()))
          .withColumn("onset_qualifier",                          F.col("onset_qualifier").cast(T.StringType()))
          .withColumn("sex_qualifier",                            F.col("sex_qualifier").cast(T.StringType()))
          # Provenance
          .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source"), ROBOKOP_SEPARATOR))
          .withColumn("publications",                             F.split(F.col("publications"), ROBOKOP_SEPARATOR))
          .withColumn("upstream_data_source",                     F.array(F.lit("robokop")))
          # Quantitative attributes
          .withColumn("num_references",                           F.lit(None).cast(T.IntegerType()))
          .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType()))
          .withColumn("has_confidence_score",                     F.col("Combined_score").cast(T.FloatType()))  # From STRING-DB
          .withColumn("extraction_confidence_score",              F.col("tmkp_confidence_score").cast(T.FloatType()))  # From Text-mining Knowledge Provider
          .withColumn("affinity",                                 F.col("affinity").cast(T.FloatType()))  # Affinity measurement type (ie pKd, pIC50, pKs)
          .withColumn("affinity_parameter",                       F.col("affinity_parameter").cast(T.StringType()))  # Affinity measurement value
          .withColumn("supporting_study_method_type",             F.col("detection_method").cast(T.StringType()))  # From IntAct
    )
    # fmt: on
    return df


def transform_edges_c5ec1f282158182f(edges_df: ps.DataFrame):
    # fmt: off
    df = (edges_df
          # Core — legacy version uses annotated column names
          .withColumnRenamed("subject:START_ID",                  "subject")
          .withColumnRenamed("predicate:TYPE",                    "predicate")
          .withColumnRenamed("object:END_ID",                     "object")
          # Qualifiers — rename annotated source cols; columns absent from source are null
          .withColumnRenamed("qualified_predicate:string",        "qualified_predicate")
          .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
          .withColumn("subject_part_qualifier",                   F.lit(None).cast(T.StringType()))
          .withColumnRenamed("object_aspect_qualifier:string",    "object_aspect_qualifier")
          .withColumnRenamed("object_direction_qualifier:string", "object_direction_qualifier")
          .withColumn("object_specialization_qualifier",          F.lit(None).cast(T.StringType()))
          .withColumn("object_part_qualifier",                    F.lit(None).cast(T.StringType()))
          .withColumnRenamed("species_context_qualifier:string",  "species_context_qualifier")
          .withColumn("disease_context_qualifier",                F.lit(None).cast(T.StringType()))
          .withColumnRenamed("frequency_qualifier:string",        "frequency_qualifier")
          .withColumn("qualifiers",                               F.lit(None).cast(T.StringType()))
          .withColumnRenamed("stage_qualifier:string",            "stage_qualifier")
          .withColumn("anatomical_context_qualifier",             F.lit(None).cast(T.StringType()))
          .withColumnRenamed("onset_qualifier:string",            "onset_qualifier")
          .withColumnRenamed("sex_qualifier:string",              "sex_qualifier")
          # Provenance — legacy version uses annotated column names
          .withColumnRenamed("knowledge_level:string",            "knowledge_level")
          .withColumnRenamed("agent_type:string",                 "agent_type")
          .withColumnRenamed("primary_knowledge_source:string",   "primary_knowledge_source")
          .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source:string[]"), ROBOKOP_SEPARATOR))
          .withColumn("publications",                             F.split(F.col("publications:string[]"), ROBOKOP_SEPARATOR))
          .withColumn("upstream_data_source",                     F.array(F.lit("robokop")))
          # Quantitative attributes
          .withColumn("num_references",                           F.lit(None).cast(T.IntegerType()))
          .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType()))
          .withColumn("has_confidence_score",                     F.col("Combined_score:string").cast(T.FloatType()))
          .withColumn("extraction_confidence_score",              F.col("tmkp_confidence_score:float").cast(T.FloatType()))
          .withColumn("affinity",                                 F.col("supporting_affinities:float[affinity:float").cast(T.FloatType()))
          .withColumnRenamed("affinity_parameter:string",         "affinity_parameter")
          .withColumn("supporting_study_method_type",             F.col("detection_method:string").cast(T.StringType()))
          )
    # fmt: on
    return df
