import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

from matrix.pipelines.integration.filters import determine_most_specific_category

from .transformer import GraphTransformer

# FUTURE: We should likely not need to rename these columns as we do below
# However, KGX is currently not as performant as we need it to be thus
# we do it manually with spark. This ought to be improved, e.g. by
# adding parquet support to KGX.
# or implementing a custom KGX version that leverages spark for higher performance
# https://github.com/everycure-org/matrix/issues/474

ROBOKOP_SEPARATOR = "\x1f"


class RobokopTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, biolink_categories_df: pd.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.
            biolink_categories_df: Biolink categories DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            nodes_df
            .withColumn("upstream_data_source",              F.array(F.lit("robokop")))
            .withColumn("all_categories",                    F.split(F.col("category:LABEL"), ROBOKOP_SEPARATOR))
            .withColumn("equivalent_identifiers",            F.split(F.col("equivalent_identifiers:string[]"), ROBOKOP_SEPARATOR))
            .withColumn("labels",                            F.array(F.col("all_categories")))
            .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", F.lit(None))
            .withColumnRenamed("id:ID", "id")
            .withColumnRenamed("name:string", "name")
            .withColumnRenamed("description:string", "description")
            # getting most specific category
            .transform(determine_most_specific_category, biolink_categories_df)
        )
        # fmt: on

    def transform_edges(self, edges_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop edges to our target schema.

        Args:
            edges_df: Edges DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            edges_df
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
        # fmt: on
