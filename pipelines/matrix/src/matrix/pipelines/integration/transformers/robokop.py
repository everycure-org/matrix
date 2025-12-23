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
          .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source"), ROBOKOP_SEPARATOR))
          .withColumn("publications",                             F.split(F.col("publications"), ROBOKOP_SEPARATOR))
          .withColumn("upstream_data_source",                     F.array(F.lit("robokop")))
          .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
          .withColumn("num_references",                           F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
          .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
    )
    # fmt: off
    return df


def transform_edges_c5ec1f282158182f(edges_df: ps.DataFrame):
    # fmt: off
    df = (edges_df
          .withColumnRenamed("subject:START_ID",                  "subject")
          .withColumnRenamed("predicate:TYPE",                    "predicate")
          .withColumnRenamed("object:END_ID",                     "object")
          .withColumnRenamed("knowledge_level:string",            "knowledge_level")
          .withColumnRenamed("primary_knowledge_source:string",   "primary_knowledge_source")
          .withColumnRenamed("object_aspect_qualifier:string",    "object_aspect_qualifier")
          .withColumnRenamed("object_direction_qualifier:string", "object_direction_qualifier")
          .withColumn("upstream_data_source",                     F.array(F.lit("robokop")))
          .withColumn("publications",                             F.split(F.col("publications:string[]"), ROBOKOP_SEPARATOR))
          .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source:string[]"), ROBOKOP_SEPARATOR))
          .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
    )
    # fmt: off
    return df
