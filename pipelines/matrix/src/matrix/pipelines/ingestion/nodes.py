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
