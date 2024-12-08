import pandas as pd
import pandera.pyspark as pa
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

from matrix.pipelines.integration.filters import determine_most_specific_category
from matrix.schemas.knowledge_graph import KGEdgeSchema, cols_for_schema, KGNodeSchema


SEPARATOR = "\x1f"


class SpokeTransformer(GraphTransformer):
    @pa.check_output(KGNodeSchema)
    def transform_nodes(self, nodes_df: DataFrame, biolink_categories_df: pd.DataFrame, **kwargs) -> DataFrame:
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
            .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers"), SEPARATOR))
            .withColumn("labels",                            F.array(F.col("all_categories")))
            .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", F.lit(None))
            # getting most specific category
            .transform(determine_most_specific_category, biolink_categories_df)
            .select(*cols_for_schema(KGNodeSchema))
        )
        # fmt: on

    @pa.check_output(KGEdgeSchema)
    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
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
            .withColumn("publications",                             F.split(F.col("publications"), SEPARATOR))
            .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source"), SEPARATOR))
            .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
            # final selection of columns
            .select(*cols_for_schema(KGEdgeSchema))
        )
        # fmt: on
