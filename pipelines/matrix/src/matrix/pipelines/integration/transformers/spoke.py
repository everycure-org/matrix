import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

from matrix.pipelines.integration.filters import determine_most_specific_category

from .transformer import GraphTransformer

SEPARATOR = "\x1f"


class SpokeTransformer(GraphTransformer):
    def __init__(self, version: str, select_cols: str = True):
        super().__init__(select_cols)
        self._version = version

    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Spoke nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.
            biolink_categories_df: Biolink categories DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "V5.2":
                df = latest_nodes_dataframe(nodes_df)
            case _:
                df = latest_nodes_dataframe(nodes_df)
        return df

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Spoke edges to our target schema.

        Args:
            edges_df: Edges DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "V5.2":
                df = latest_edges_dataframe(edges_df)
            case _:
                df = latest_edges_dataframe(edges_df)
        return df


def latest_nodes_dataframe(nodes_df: ps.DataFrame):
    # fmt: off
    df = (
        nodes_df
        .withColumn("upstream_data_source",              F.array(F.lit("spoke")))
        .withColumn("all_categories",                    F.split(F.col("category"), SEPARATOR))
        .withColumn("equivalent_identifiers",            F.lit(None).cast(T.ArrayType(T.StringType())))
        .withColumn("labels",                            F.lit(None).cast(T.ArrayType(T.StringType())))
        .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
        .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
        # getting most specific category
        .transform(determine_most_specific_category)
    )
    # fmt: on
    return df


def latest_edges_dataframe(edges_df: ps.DataFrame):
    # fmt: off
    df = (edges_df
          .withColumn("knowledge_level",                          F.lit(None).cast(T.StringType()))
          .withColumn("agent_type",                               F.lit(None).cast(T.StringType()))
          .withColumn("aggregator_knowledge_source",              F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("publications",                             F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("upstream_data_source",                     F.array(F.lit("spoke")))
          .withColumn("primary_knowledge_source",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
          .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
          .withColumn("num_references",                           F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
          .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
          .withColumn("object_aspect_qualifier",                  F.lit(None).cast(T.StringType()))
          .withColumn("object_direction_qualifier",               F.lit(None).cast(T.StringType()))
          )
    # fmt: on
    return df
