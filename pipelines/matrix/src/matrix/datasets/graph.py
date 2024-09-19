"""Graph module.

Module containing knowledge graph representation and utilities.
"""
import pandas as pd
import logging

from kedro_datasets.pandas import ParquetDataset

from typing import Any, Dict, List
from kedro.io.core import Version


logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Class to represent a knowledge graph.

    NOTE: Provide handover point to Neo4J in the future.
    """

    def __init__(self, nodes: pd.DataFrame) -> None:
        """Initializes the KnowledgeGraph instance.

        Args:
            nodes: DataFrame containing nodes of the graph.
        """
        self._nodes = nodes
        self._node_index = dict(zip(nodes["id"], nodes.index))

        # Add type specific indexes
        self._drug_nodes = list(nodes[nodes["is_drug"]]["id"])
        self._disease_nodes = list(nodes[nodes["is_disease"]]["id"])
        self._embeddings = dict(zip(nodes["id"], nodes["topological_embedding"]))

    def get_embedding(self, node_id: str, default: Any = None):
        """Retrieves embedding for node with the ID.

        Args:
            node_id: Node ID.
            default: default value to return
        Returns:
            Embedding or None if not found
        """
        res = self._embeddings.get(node_id, default)
        if res is default:
            logger.warning(f"Embedding for node with id '{node_id}' not found!")

        return res

    def flags_to_ids(self, flags: List[str]) -> List[str]:
        """Helper function for extracting nodes from flag columns.

        Args:
            flags: List of names of boolean columns in the graph nodes.

        Returns:
            List of graph nodes id's satisfying all flags.
        """
        is_all_flags = self._nodes[flags].all(axis=1)
        select_nodes = self._nodes[is_all_flags]
        return list(select_nodes["id"])

    def get_node_attribute(self, node_id: str, col_name: str) -> any:
        """Retrieves chosen node attributes from the ID.

        Args:
            node_id: Node ID.
            col_name: Name of column containing desired attribute
        """
        return self._nodes[self._nodes["id"] == node_id][col_name]


class KnowledgeGraphDataset(ParquetDataset):
    """Dataset adaptor to read KnowledgeGraph using Kedro's dataset functionality."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Initializes the KnowledgeGraphDataset."""
        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
            metadata=metadata,
            **kwargs,
        )

    def _load(self) -> KnowledgeGraph:
        return KnowledgeGraph(super()._load())
