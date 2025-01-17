from typing import Collection, Type

import pandera
import pyspark.sql as ps
from pyspark.sql.functions import array_join

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

SEPARATOR = "\x1f"


def join_array_columns(df: ps.DataFrame, cols: Collection[str], sep: str = SEPARATOR) -> ps.DataFrame:
    for c in cols:
        df = df.withColumn(c, array_join(c, delimiter=sep))
    return df


def create_kgx_format(df: ps.DataFrame, cols: Collection[str], model: Type[pandera.DataFrameModel]):
    return join_array_columns(df, cols=cols).select(*cols_for_schema(model))


def filtered_edges_to_kgx(df: ps.DataFrame) -> ps.DataFrame:
    """Function to create KGX formatted edges.

    Args:
        df: Edges dataframe
    """
    return create_kgx_format(
        df,
        cols=("upstream_data_source", "aggregator_knowledge_source", "publications"),
        model=KGEdgeSchema,
    )


def filtered_nodes_to_kgx(df: ps.DataFrame) -> ps.DataFrame:
    """Function to create KGX formatted nodes.

    Args:
        df: Nodes dataframe
    """
    return create_kgx_format(
        df,
        cols=("equivalent_identifiers", "all_categories", "publications", "labels", "upstream_data_source"),
        model=KGNodeSchema,
    )
