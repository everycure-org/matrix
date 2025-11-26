from typing import List

import pyspark.sql as ps
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql import types as T
from pyspark.sql.functions import array_join

SEPARATOR = "\x1f"


def join_array_columns(df: ps.DataFrame, cols: List[str], sep: str = SEPARATOR) -> ps.DataFrame:
    for c in cols:
        df = df.withColumn(c, array_join(c, delimiter=sep))
    return df


@check_output(
    DataFrameSchema(
        columns={
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False),
            "object": Column(T.StringType(), nullable=False),
            "upstream_data_source": Column(T.StringType(), nullable=False),
            "aggregator_knowledge_source": Column(T.StringType(), nullable=True),
            "publications": Column(T.StringType(), nullable=True),
        },
        unique=["subject", "predicate", "object"],
    ),
    pass_columns=True,
)
def unified_edges_to_kgx(df: ps.DataFrame, cols: List[str]) -> ps.DataFrame:
    """Function to create KGX formatted edges.

    Args:
        df: Edges dataframe
    """
    return df.transform(
        join_array_columns, cols=["upstream_data_source", "aggregator_knowledge_source", "publications"]
    ).select(*cols)


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "category": Column(T.StringType(), nullable=False),
            "description": Column(T.StringType(), nullable=True),
            "equivalent_identifiers": Column(T.StringType(), nullable=False),
            "all_categories": Column(T.StringType(), nullable=False),
            "publications": Column(T.StringType(), nullable=True),
            "labels": Column(T.StringType(), nullable=False),
            "upstream_data_source": Column(T.StringType(), nullable=False),
        },
        unique=["id"],
    ),
    pass_columns=True,
)
def unified_nodes_to_kgx(df: ps.DataFrame, cols: List[str]) -> ps.DataFrame:
    """Function to create KGX formatted nodes.

    Args:
        df: Nodes dataframe
    """
    return df.transform(
        join_array_columns,
        cols=["equivalent_identifiers", "all_categories", "publications", "labels", "upstream_data_source"],
    ).select(*cols)
