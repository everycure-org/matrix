import logging
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import GraphTransformer

logger = logging.getLogger(__name__)


class ClinicalTrailsTransformer(GraphTransformer):
    # @pa.check_output(KGNodeSchema)
    def transform_nodes(self, nodes_df: DataFrame, **kwargs) -> DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        df = (
            nodes_df
            .withColumn("id",                                f.lit("normalized_curie"))
            .withColumn("upstream_data_source",              f.array(f.lit("ec_clinical_trails")))
            .withColumn("labels",                            f.array(f.lit("entity label"))) # TODO: Fix entity labels for medical?
            .withColumn("all_categories",                    f.array(f.lit("biolink:"))) # TODO fix
            .withColumn("equivalent_identifiers",            f.array(f.col("id")))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.col("id"))
            # .transform(determine_most_specific_category, biolink_categories_df) need this?
            # Filter nodes we could not correctly resolve
            .filter(f.col("id").isNotNull())
            # .select(*cols_for_schema(KGNodeSchema))
        )
        return df
        # fmt: on

    # @pa.check_output(KGEdgeSchema)
    def transform_edges(self, edges_df: DataFrame, **kwargs) -> DataFrame:
        """Transform edges to our target schema.

        Args:
            edges_df: Edges DataFrame.
            pubmed_mapping: pubmed mapping
        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        return (
            edges_df
            .withColumn("subject",                       f.lit("drug_kg_curie"))
            .withColumn("object",                        f.lit("disease_kg_curie"))
            .filter((f.col("subject").isNotNull()) & (f.col("object").isNotNull()))
        )
