import pandas as pd
import pandera.pyspark as pa
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql as ps

from .transformer import GraphTransformer

from matrix.pipelines.integration.filters import determine_most_specific_category
from matrix.schemas.knowledge_graph import KGEdgeSchema, cols_for_schema, KGNodeSchema


SEPARATOR = "\x1f"


class SpokeTransformer(GraphTransformer):
    @pa.check_output(KGNodeSchema)
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
            .withColumn("equivalent_identifiers",            F.lit(None))
            .withColumn("labels",                            F.lit(None))
            .withColumn("publications",                      F.lit(None))
            .withColumn("international_resource_identifier", F.lit(None))
            # getting most specific category
            .transform(determine_most_specific_category, biolink_categories_df)
            .select(*cols_for_schema(KGNodeSchema))
        )
        # fmt: on

    @pa.check_output(KGEdgeSchema)
    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Spoke edges to our target schema.

        Args:
            edges_df: Edges DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            edges_df
            .withColumn("upstream_data_source",                     F.array(F.lit("spoke")))
            .withColumn("publications",                             F.lit(None))
            .withColumn("knowledge_level",                          F.lit(None))
            .withColumn("primary_knowledge_source",                 F.lit(None))
            .withColumn("aggregator_knowledge_source",              F.lit(None))
            .withColumn("object_aspect_qualifier",                  F.lit(None))
            .withColumn("object_direction_qualifier",               F.lit(None))
            .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
            # final selection of columns
            .select(*cols_for_schema(KGEdgeSchema))
        )
        # fmt: on
