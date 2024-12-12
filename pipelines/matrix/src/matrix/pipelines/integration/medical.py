import logging
import pandas as pd
import pandera.pyspark as pa
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from .transformer import GraphTransformer
from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema
from matrix.pipelines.integration.filters import determine_most_specific_category

logger = logging.getLogger(__name__)


class MedicalTransformer(GraphTransformer):
    @pa.check_output(KGNodeSchema)
    def transform_nodes(self, nodes_df: DataFrame, biolink_categories_df: pd.DataFrame, **kwargs) -> DataFrame:
        """Transform nodes to our target schema.

        Args:
            nodes_df: Nodes DataFrame.

        Returns:
            Transformed DataFrame.
        """
        # fmt: off
        df = (
            nodes_df
            .withColumn("labels",                            f.array(f.col("name")))
            .withColumn("all_categories",                    f.array(f.col("category")))
            .withColumn("equivalent_identifiers",            f.array(f.col("id")))
            .withColumn("publications",                      f.lit(None).cast(T.ArrayType(T.StringType())))
            .withColumn("international_resource_identifier", f.col("id"))
            .transform(determine_most_specific_category, biolink_categories_df)
            .select(*cols_for_schema(KGNodeSchema))
        )
        # SILC Fix 
        chemical_ids = ['OMIM:MTHU008082', 'UMLS:C0311400', 'EFO:0004501', 
                    'LOINC:LP14446-6', 'OMIM:MTHU000104', 'LOINC:LP89782-4']

        # Add ChemicalEntity category for specific IDs
        df = df.withColumn(
            "all_categories",
            f.when(
                f.col("id").isin(chemical_ids),
                f.array_union(f.col("all_categories"), f.array(f.lit("biolink:Case")))
            ).otherwise(f.col("all_categories"))
        ).withColumn(
            "category",
            f.when(
                f.col("id").isin(chemical_ids),
                f.lit("biolink:Case")
            ).otherwise(f.col("category"))
        )
        return df
        # fmt: on

    @pa.check_output(KGEdgeSchema)
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
            .withColumn("knowledge_level",               f.lit(None).cast(T.StringType()))
            .withColumn("aggregator_knowledge_source",   f.array(f.col("knowledge_source")))
            .withColumn("primary_knowledge_source",      f.lit('medical team').cast(T.StringType()))
            .withColumn("publications",                  f.array(f.lit('medical team')))
            .withColumn("subject_aspect_qualifier",      f.lit(None).cast(T.StringType())) #not present
            .withColumn("subject_direction_qualifier",   f.lit(None).cast(T.StringType())) #not present
            .withColumn("object_aspect_qualifier",       f.lit(None).cast(T.StringType())) #not present
            .withColumn("object_direction_qualifier",    f.lit(None).cast(T.StringType())) #not present
            .select(*cols_for_schema(KGEdgeSchema))
        )
