import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from matrix.schemas.knowledge_graph import cols_for_schema, KGNodeSchema, KGEdgeSchema

logger = logging.getLogger(__name__)


def filtered_edges_to_kgx(df: DataFrame) -> DataFrame:
    """Function to create KGX formatted edges.

    Args:
        df: Edges dataframe
    """
    return (
        df
        .withColumn("upstream_data_source", F.array_join(F.col("upstream_data_source"), ","))
        .withColumn("aggregator_knowledge_source", F.array_join(F.col("aggregator_knowledge_source"), ","))
        .withColumn("publications", F.array_join(F.col("publications"), ","))
        # .select("subject", "predicate", "object", "knowledge_level", "primary_knowledge_source", "object_aspect_qualifier", "object_direction_qualifier", "upstream_data_source", "publications", "aggregator_knowledge_source", "subject_aspect_qualifier", "subject_direction_qualifier")
        .select(*cols_for_schema(KGEdgeSchema))
    )


def filtered_nodes_to_kgx(df: DataFrame) -> DataFrame:
    """Function to create KGX formatted nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df
        .withColumn("equivalent_identifiers", F.array_join(F.col("equivalent_identifiers"), ","))
        .withColumn("all_categories", F.array_join(F.col("all_categories"), ","))
        .withColumn("publications", F.array_join(F.col("publications"), ","))
        .withColumn("labels", F.array_join(F.col("labels"), ","))
        .withColumn("upstream_data_source", F.array_join(F.col("upstream_data_source"), ","))
        #.select("id", "name", "category", "description", "equivalent_identifiers", "all_categories", "publications", "labels", "international_resource_identifier", "upstream_data_source")
        .select(*cols_for_schema(KGNodeSchema))
    )
