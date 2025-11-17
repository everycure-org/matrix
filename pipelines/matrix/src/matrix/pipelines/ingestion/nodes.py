import pandas as pd
import pandera.pandas as pa

import polars as pl
from matrix_io_utils.robokop import (
    robokop_convert_boolean_columns_to_label_columns,
    robokop_strip_type_from_column_names,
)


def robokop_preprocessing_nodes(nodes: pl.LazyFrame) -> pl.DataFrame:
    """Build the nodes tsv file.

    Args:
    nodes: The nodes.tsv file from a Robokop download
    """
    nodes = robokop_convert_boolean_columns_to_label_columns(nodes)
    nodes = robokop_strip_type_from_column_names(nodes.lazy())
    return nodes


def robokop_preprocessing_edges(edges: pl.LazyFrame) -> pl.DataFrame:
    """Build the edges tsv file.

    Args:
    nodes: The edges.tsv file from a Robokop download
    """
    return robokop_strip_type_from_column_names(edges)

# FUTURE: make schema checks dynamic per transform function
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    )
                ],
            ),
            "translator_id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="translator_id must be unique",
                    )
                ],
            ),
            "deleted": pa.Column(bool, nullable=False, checks=[pa.Check(lambda col: col == False)]),
        },
    )
)
def write_drug_list(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["deleted"]]


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    )
                ],
            ),
            "deleted": pa.Column(bool, nullable=False, checks=[pa.Check(lambda col: col == False)]),
        },
    )
)
def write_disease_list(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["deleted"]]

