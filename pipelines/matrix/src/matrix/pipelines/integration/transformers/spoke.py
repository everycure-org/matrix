import pandas as pd
import pandera.pyspark as pa
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

from matrix.pipelines.integration import schema
from matrix.pipelines.integration.filters import determine_most_specific_category

from .transformer import GraphTransformer

SEPARATOR = "\x1f"


class SpokeTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, biolink_categories_df: pd.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Spoke nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.
            biolink_categories_df: Biolink categories DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            nodes_df
            .withColumn("upstream_data_source",              F.array(F.lit("spoke")))
            .withColumn("all_categories",                    F.split(F.col("category"), SEPARATOR))
            .withColumn("equivalent_identifiers",            F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("labels",                            F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
            .select(*[col for col in schema.BIOLINK_KG_NODE_SCHEMA.columns.keys()])
        )
        # fmt: on

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Spoke edges to our target schema.

        Args:
            edges_df: Edges DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return  (
            edges_df
            .withColumn("upstream_data_source",                     F.array(F.lit("spoke")))
            .withColumn("publications",                             F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("knowledge_level",                          F.lit(None).cast(T.StringType()))
            .withColumn("primary_knowledge_source",                 F.lit(None).cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",              F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("object_aspect_qualifier",                  F.lit(None).cast(T.StringType()))
            .withColumn("object_direction_qualifier",               F.lit(None).cast(T.StringType()))
            .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
            # final selection of columns
            .select(*[col for col in schema.BIOLINK_KG_EDGE_SCHEMA.columns.keys()])
        )
        # fmt: on
