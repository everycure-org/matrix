import logging
from typing import Any, Dict, List

import pandas as pd
from kedro.io.core import (
    DatasetError,
    Version,
)
from kedro_datasets.pandas import ParquetDataset

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Class to represent a knowledge graph.

    FUTURE: We should aspire to remove this class
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
        self._rtxkg2_embeddings = dict(zip(nodes["id"], nodes["rtxkg2_topological_embedding"]))
        self._robokop_embeddings = dict(zip(nodes["id"], nodes["robokop_topological_embedding"]))

    def get_embedding(self, node_id: str, default: Any = None):
        """Retrieves embedding for node with the ID.
        Args:
            node_id: Node ID.
            default: default value to return
        Returns:
            Embedding or None if not found
        """
        return {"rtxkg2": self._rtxkg2_embeddings[node_id], "robokop": self._robokop_embeddings[node_id]}

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


class PandasParquetDataset(ParquetDataset):
    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._as_type = None
        if load_args is not None:
            self._as_type = load_args.pop("as_type", None)

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
            metadata=metadata,
        )

    def save(self, df: pd.DataFrame):
        return super().save(df)

    def load(self) -> KnowledgeGraph:
        attempt = 0

        # Retrying due to a very flaky error, causing GCS not retrieving
        # parquet files on first try.
        while attempt < 3:
            try:
                # Attempt reading the object
                # https://github.com/everycure-org/matrix/issues/71
                df = super().load()

                if self._as_type:
                    return df.astype(self._as_type)

                return df
            except Exception:
                attempt += 1
                logger.warning(f"Parquet file `{self._filepath}` not found, retrying!")

        raise DatasetError(f"Unable to find the Parquet file `{self._filepath}` underlying this dataset!")


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

    def load(self) -> KnowledgeGraph:
        attempt = 0

        # Retrying due to a very flaky error, causing GCS not retrieving
        # parquet files on first try.
        while attempt < 3:
            try:
                # Attempt reading the object
                # https://github.com/everycure-org/matrix/issues/71
                return KnowledgeGraph(super().load())
            except Exception:
                attempt += 1
                logger.warning(f"Parquet file `{self._filepath}` not found, retrying!")

        raise DatasetError(f"Unable to find the Parquet file `{self._filepath}` underlying this dataset!")
