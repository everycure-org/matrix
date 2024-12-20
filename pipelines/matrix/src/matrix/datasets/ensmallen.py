from pathlib import PurePosixPath
from typing import Any, Dict
from copy import deepcopy

import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
import ensmallen
import pathlib
import pyarrow.dataset as ds
import pyarrow as pa
import logging

logger = logging.getLogger(__name__)

# TODO: Hack: need to overwrite ensmallen.Graph.__str__ (very expensive execution)
# since it gets called by the pipeline somehow
ensmallen.Graph.__str__ = lambda self: "monkey patched ensmallen.Graph.__str__"
ensmallen.Graph.__repr__ = lambda self: "monkey patched ensmallen.Graph.__repr__"

# general naming conventions in EC-pipeline
NODE_ID = "id"
NODE_TYPE = "category"
EDGE_SRC = "subject"
EDGE_DST = "object"
EDGE_TYPE = "predicate"


class GraphDataset(AbstractDataset[ensmallen.Graph, ensmallen.Graph]):
    """ensmallen Graph, built on a nodes.csv and edges.csv file."""

    def __init__(self, filepath: str):
        """Creates a new instance of GraphDataset to load / save a ensmallen.Graph for given filepath.
        `
        Args:
            filepath: The location of the ensmallens file to load / save data.
                      Convention is: filepath is a directory with two files: `nodes.tsv` and `edges.tsv
        """
        # parse the path and protocol (e.g. file, http, s3, etc.) for nodes
        protocol_nodes, path_nodes = get_protocol_and_path(PurePosixPath(filepath) / "nodes.tsv")
        self._protocol_nodes = protocol_nodes
        self._filepath_nodes = PurePosixPath(path_nodes)
        self._fs_nodes = fsspec.filesystem(self._protocol_nodes)

        # parse the path and protocol (e.g. file, http, s3, etc.) for edges
        protocol_edges, path_edges = get_protocol_and_path(PurePosixPath(filepath) / "edges.tsv")
        self._protocol_edges = protocol_edges
        self._filepath_edges = PurePosixPath(path_edges)
        self._fs_edges = fsspec.filesystem(self._protocol_edges)

    def _load(self) -> np.ndarray:
        """load ensmallens graph from disk"""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path_nodes = get_filepath_str(self._filepath_nodes, self._protocol_nodes)
        load_path_edges = get_filepath_str(self._filepath_edges, self._protocol_edges)
        logger.debug(f"Loading ensmallen from {load_path_nodes} and {load_path_edges}!")

        g = ensmallen.Graph.from_csv(
            directed=False,
            node_path=load_path_nodes,
            nodes_column=NODE_ID,
            node_list_node_types_column=NODE_TYPE,
            edge_path=load_path_edges,
            edge_list_edge_types_column=EDGE_TYPE,
            default_weight=1,
        )

        logger.debug("Done loading!")
        return g

    def _save(self, G: ensmallen.Graph) -> None:
        """Saves ensmallen.Graph to the specified filepath."""

        assert self._filepath_nodes.parent == self._filepath_edges.parent, "nodes and edges in two different folders!"
        parent_folder = pathlib.Path(self._filepath_nodes.parent)
        if not parent_folder.exists():
            parent_folder.mkdir()

        save_path_nodes = get_filepath_str(self._filepath_nodes, self._protocol_nodes)
        save_path_edges = get_filepath_str(self._filepath_edges, self._protocol_edges)

        logger.debug(f"Saving ensmallen nodes to {save_path_nodes}")

        G.dump_nodes(
            save_path_nodes,
            verbose=True,
            header=True,
            nodes_column_number=0,
            nodes_column=NODE_ID,
            node_types_column_number=1,
            node_type_column=NODE_TYPE,
        )

        logger.debug(f"Saving ensmallen edges to {save_path_nodes}")
        G.dump_edges(
            save_path_edges,
            verbose=True,
            header=True,
            edge_types_column_number=2,
            edge_type_column=EDGE_TYPE,
        )

    def _describe(self) -> Dict[str, Any]:
        return {"summy": "dummy"}


class GraphDatasetArrow(AbstractDataset[ensmallen.Graph, ensmallen.Graph]):
    """ensmallen.Graph backed by on-disk nodes/edges dataframes (parquet)"""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {
        "nodename_col": NODE_ID,
        "nodetype_col": NODE_TYPE,
        "edge_src_col": EDGE_SRC,
        "edge_dst_col": EDGE_DST,
        "edge_type_col": EDGE_TYPE,
    }
    DEFAULT_SAVE_ARGS: dict[str, Any] = {
        "nodename_col": NODE_ID,
        "nodetype_col": NODE_TYPE,
        "edge_src_col": EDGE_SRC,
        "edge_dst_col": EDGE_DST,
        "edge_type_col": EDGE_TYPE,
    }

    def __init__(self, filepath: str, load_args=None, save_args=None):
        """Creates a new instance of GraphDataset to load / save a ensmallen.Graph for given filepath.
        `
        Args:
            filepath: The location of the ensmallens file to load / save data.
                      Convention is: filepath is a directory with two files: `nodes.tsv` and `edges.tsv
        """
        # parse the path and protocol (e.g. file, http, s3, etc.) for nodes
        protocol_nodes, path_nodes = get_protocol_and_path(
            PurePosixPath(filepath) / "nodes"
        )  # this better be a parquet folder
        self._protocol_nodes = protocol_nodes
        self._filepath_nodes = PurePosixPath(path_nodes)
        self._fs_nodes = fsspec.filesystem(self._protocol_nodes)

        # parse the path and protocol (e.g. file, http, s3, etc.) for edges
        protocol_edges, path_edges = get_protocol_and_path(
            PurePosixPath(filepath) / "edges"
        )  # this better be a parquet folder
        self._protocol_edges = protocol_edges
        self._filepath_edges = PurePosixPath(path_edges)
        self._fs_edges = fsspec.filesystem(self._protocol_edges)

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path_nodes = get_filepath_str(self._filepath_nodes, self._protocol_nodes)
        load_path_edges = get_filepath_str(self._filepath_edges, self._protocol_edges)
        logger.debug(f"Loading ensmallen from {load_path_nodes} and {load_path_edges} via arrow!")

        g = ensmallen.Graph.from_parquet(
            load_path_nodes,
            load_path_edges,
            nodename_col=self._load_args["nodename_col"],  # id
            nodetype_col=self._load_args["nodetype_col"],  #'category',
            edge_src_col=self._load_args["edge_src_col"],  #'subject',
            edge_dst_col=self._load_args["edge_dst_col"],  #'object',
            edge_type_col=self._load_args["edge_type_col"],  #'predicate'
        )

        logger.debug("Done loading!")
        return g

    def _save(self, G: ensmallen.Graph) -> None:
        """Saves ensmallen.Graph to the specified filepath."""

        # BTW: no need to monkey patch the __str__ here, its not called
        # after _save finishes
        assert self._filepath_nodes.parent == self._filepath_edges.parent, "nodes and edges in two different folders!"
        parent_folder = pathlib.Path(self._filepath_nodes.parent)
        if not parent_folder.exists():
            parent_folder.mkdir()

        save_path_nodes = get_filepath_str(self._filepath_nodes, self._protocol_nodes)
        save_path_edges = get_filepath_str(self._filepath_edges, self._protocol_edges)

        logger.debug(f"Saving ensmallen nodes to {save_path_nodes}")

        nodes_table, edges_table = ensmallen_to_pq_datasets(
            G,
            nodename_col=self._save_args["nodename_col"],  # id
            nodetype_col=self._save_args["nodetype_col"],  #'category',
            edge_src_col=self._save_args["edge_src_col"],  #'subject',
            edge_dst_col=self._save_args["edge_dst_col"],  #'object',
            edge_type_col=self._save_args["edge_type_col"],  #'predicate'
        )

        logger.info(f"writing nodes to {save_path_nodes}")
        ds.write_dataset(nodes_table, save_path_nodes, format="parquet", existing_data_behavior="delete_matching")

        logger.info(f"writing edges to {save_path_edges}")
        ds.write_dataset(edges_table, save_path_edges, format="parquet", existing_data_behavior="delete_matching")

    def _describe(self) -> Dict[str, Any]:
        return {"dummy": "dummy"}


def ensmallen_to_pq_datasets(
    G: ensmallen.Graph,
    nodename_col,
    nodetype_col,
    edge_src_col,
    edge_dst_col,
    edge_type_col,
):
    """turn graph into an node and edge arrow.Table"""

    node_ids, nodetypes = _get_nodes(G)

    edges = _get_edges(G, directed=False)  # TODO check directed
    sub, pred, obj = zip(*edges)

    nodes_table = pa.table(
        [pa.array(node_ids, type=pa.string()), pa.array(nodetypes, type=pa.string())],
        names=[nodename_col, nodetype_col],
    )
    edges_table = pa.table(
        [pa.array(sub, type=pa.string()), pa.array(pred, type=pa.string()), pa.array(obj, type=pa.string())],
        names=[edge_src_col, edge_type_col, edge_dst_col],
    )

    return nodes_table, edges_table


def _get_nodes(G: ensmallen.Graph):
    """return all nodes and their type"""
    nodes = G.get_node_names()
    node_types = [_[0] for _ in G.get_node_type_names()]  # TODO some weirdness: categories are lists
    return nodes, node_types


def _get_edges(G: ensmallen.Graph, directed: bool):
    """get a list of tuples: subject,predicate,object
    from the graph
    """
    # we need tuples of subject, predicate, object
    # but ensmallen stores it very awkwaredly
    # need to iterate over edge types to get the types
    nodes = G.get_node_names()

    edges = []
    edges_types = G.get_edge_type_names_counts_hashmap()
    for etype in edges_types.keys():
        source_target = G.get_edge_node_ids_from_edge_type_name(directed, etype)
        subject_id = source_target[:, 0]
        object_id = source_target[:, 1]

        subject_names = [nodes[i] for i in subject_id]
        object_names = [nodes[i] for i in object_id]
        for s, o in zip(subject_names, object_names):
            edges.append((s, etype, o))
    return edges
