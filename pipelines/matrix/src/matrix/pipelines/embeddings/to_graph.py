import pathlib
import tempfile
import ensmallen
from pyspark.sql.dataframe import DataFrame
from matrix.datasets.ensmallen import NODE_ID, NODE_TYPE, EDGE_SRC, EDGE_DST, EDGE_TYPE

import logging

logger = logging.getLogger(__name__)


# TODO: __str__ on Graph is VERY expensive (see GraphDataset.load)
# and it somehow called by kedro
# let monkey patch it here, so it runs fast
ensmallen.Graph.__str__ = lambda self: "monkey patched ensmallen.Graph.__str__"
ensmallen.Graph.__repr__ = lambda self: "monkey patched ensmallen.Graph.__repr__"


def check_column_is_unique(df: DataFrame, column_name: str) -> bool:
    unique_counts = df.select(column_name).distinct().count()
    original_counts = df.count()
    return original_counts == unique_counts


def convert_parquet_to_ensmallen(nodes_df: DataFrame, edges_df: DataFrame) -> ensmallen.Graph:
    """
    create ensmallen.Graph from pySpark, reading the pyspark.Dataframes directly from disk via `ensmallen.Graph.from_parquet`

    TODO: a hack; we really need the underlying filenames to feed into the graph constructor (.from_parquet),
    rather than the instantiated DataFrames, but its hard to get that out of kedro.
    Instead: write nodes_df/edges_df to a known (temporary) filename and put those into `from_parquet()`
    """
    assert check_column_is_unique(nodes_df, NODE_ID), "node ids not unique!!"

    with tempfile.TemporaryDirectory() as save_path:
        save_path_nodes = pathlib.Path(save_path) / "nodes"
        save_path_edges = pathlib.Path(save_path) / "edges"

        logger.info(f"writing tmp nodes to {save_path_nodes}")
        nodes_df.select(NODE_ID, NODE_TYPE).write.parquet(str(save_path_nodes), mode="overwrite")

        logger.info(f"writing tmp edges to {save_path_edges}")
        edges_df.select(EDGE_SRC, EDGE_DST, EDGE_TYPE).write.parquet(str(save_path_edges), mode="overwrite")

        logger.info("constructing graph")
        g = ensmallen.Graph.from_parquet(
            str(save_path_nodes),
            str(save_path_edges),
            nodename_col=NODE_ID,
            nodetype_col=NODE_TYPE,
            edge_src_col=EDGE_SRC,
            edge_dst_col=EDGE_DST,
            edge_type_col=EDGE_TYPE,
        )

    return g
