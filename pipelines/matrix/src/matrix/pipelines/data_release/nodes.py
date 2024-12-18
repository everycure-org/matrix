import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from matrix.schemas.knowledge_graph import cols_for_schema, KGNodeSchema, KGEdgeSchema

logger = logging.getLogger(__name__)

SEPARATOR = "\x1f"


def filtered_edges_to_kgx(df: DataFrame) -> DataFrame:
    """Function to create KGX formatted edges.

    Args:
        df: Edges dataframe
    """
    return (
        df.withColumn("upstream_data_source", F.array_join(F.col("upstream_data_source"), SEPARATOR))
        .withColumn("aggregator_knowledge_source", F.array_join(F.col("aggregator_knowledge_source"), SEPARATOR))
        .withColumn("publications", F.array_join(F.col("publications"), SEPARATOR))
        .select(*cols_for_schema(KGEdgeSchema))
    )


def filtered_nodes_to_kgx(df: DataFrame) -> DataFrame:
    """Function to create KGX formatted nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.withColumn("equivalent_identifiers", F.array_join(F.col("equivalent_identifiers"), SEPARATOR))
        .withColumn("all_categories", F.array_join(F.col("all_categories"), SEPARATOR))
        .withColumn("publications", F.array_join(F.col("publications"), SEPARATOR))
        .withColumn("labels", F.array_join(F.col("labels"), SEPARATOR))
        .withColumn("upstream_data_source", F.array_join(F.col("upstream_data_source"), SEPARATOR))
        .select(*cols_for_schema(KGNodeSchema))
    )
