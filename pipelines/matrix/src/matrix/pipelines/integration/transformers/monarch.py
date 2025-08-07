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

SEPARATOR = r"\|"


class MonarchTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            nodes_df
            .withColumn("upstream_data_source",     F.array(F.lit("monarch")))
            .withColumn("all_categories",           F.split(F.col("category"), SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("equivalent_identifiers",   F.split(F.col("_xref"), SEPARATOR))
            .withColumn("labels",                   F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("publications",             F.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumnRenamed("_iri", "international_resource_identifier")
            # getting most specific category
            .transform(determine_most_specific_category)
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
            .withColumn("aggregator_knowledge_source",              F.split(F.col("aggregator_knowledge_source"), SEPARATOR))
            .withColumn("publications",                             F.split(F.col("publications"), SEPARATOR))
            .withColumn("upstream_data_source",                     F.array(F.lit("monarch")))
            .withColumn("subject_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",              F.lit(None).cast(T.StringType()))
            .withColumn("num_references",                           F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
            .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
            .withColumn("object_aspect_qualifier",                 F.lit(None).cast(T.StringType()))
            .withColumn("object_direction_qualifier",              F.lit(None).cast(T.StringType()))
        )
        # fmt: on
