import logging

import pyspark.sql as ps
import pyspark.sql.functions as f
import pyspark.sql.types as T

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class EmbiologyTransformer(GraphTransformer):
    def transform_nodes(self, nodes_df: ps.DataFrame, **kwargs) -> ps.DataFrame:
        """Transform Embiology nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        EMBIOLOGY_SEPARATOR = r"\|"
        return (
            nodes_df
            .withColumn("upstream_data_source",              f.array(f.lit("embiology")))
            .withColumn("labels",                            f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("all_categories",                    f.split(f.col("all_categories") , EMBIOLOGY_SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("equivalent_identifiers",            f.split(f.col("equivalent_identifiers"), EMBIOLOGY_SEPARATOR))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.lit(None).cast(T.StringType())) # TODO: add international resource identifier
            .withColumn("description",                       f.lit(None).cast(T.StringType()))
        )
        # fmt: on

    def transform_edges(
        self,
        edges_df: ps.DataFrame,
        **kwargs,
    ) -> ps.DataFrame:
        """Transform Embiology edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
            pubmed_mapping: pubmed mapping
        Returns:
            Transformed DataFrame.
        """

        # fmt: off
        EMBIOLOGY_SEPARATOR = r"\|"
        return (
            edges_df
            .withColumn("upstream_data_source",          f.array(f.lit("embiology")))
            .withColumn("aggregator_knowledge_source",   f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("publications",                  f.split(f.col("publications"), EMBIOLOGY_SEPARATOR).cast(T.ArrayType(T.StringType())))
            .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
            .withColumn("agent_type",                    f.lit(None).cast(T.StringType()))
            .withColumn("primary_knowledge_source",      f.lit(None).cast(T.StringType()))
            .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType()))
            .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType()))
            .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType()))
            .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType()))
            .withColumn("num_references",                f.cast(T.IntegerType(), f.col("num_references")))
            .withColumn("num_sentences",                 f.cast(T.IntegerType(), f.col("num_sentences")))
        )
        # fmt: on
