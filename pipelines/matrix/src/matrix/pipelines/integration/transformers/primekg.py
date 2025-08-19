import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T

# FUTURE: We should likely not need to rename these columns as we do below
# However, KGX is currently not as performant as we need it to be thus
# we do it manually with spark. This ought to be improved, e.g. by
# adding parquet support to KGX.
# or implementing a custom KGX version that leverages spark for higher performance
# https://github.com/everycure-org/matrix/issues/474
from .transformer import GraphTransformer

SEPARATOR = r"\|"


class PrimeKGTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Robokop nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        match self._version:
            case "2.1.1":
                df = transform_nodes_2_1_1(nodes_df)
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
            case "2.1.1":
                df = transform_edges_2_1_1(edges_df)
            case _:
                raise NotImplementedError(f"No edges transformer code implemented for version: {self._version}")
        return df


def transform_nodes_2_1_1(nodes_df: ps.DataFrame):
    # fmt: off
    df = (nodes_df
          .withColumn("upstream_data_source",              F.array(F.lit("primekg")))
          .withColumn("all_categories",                    F.split(F.col("category"), SEPARATOR))
          .withColumn("equivalent_identifiers",            F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("labels",                            F.col("all_categories"))
          .withColumn("publications",                      F.lit(None).cast(T.ArrayType(T.StringType())))
          .withColumn("international_resource_identifier", F.lit(None).cast(T.StringType()))
          )
    # fmt: on
    return df


def transform_edges_2_1_1(edges_df: ps.DataFrame):
    # fmt: off
    df = (edges_df
          .withColumn("publications",                             F.split(F.col("publications"), SEPARATOR))
          .withColumn("upstream_data_source",                     F.array(F.lit("primekg")))
          .withColumn("num_references",                           F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
          .withColumn("num_sentences",                            F.lit(None).cast(T.IntegerType())) # Required to match EmBiology schema
          )
    # fmt: off
    return df
