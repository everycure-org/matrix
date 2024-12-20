from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
import logging

import anndata

logger = logging.getLogger(__name__)

""".
AnnData: annotated dataframe, i.e. a matrix with row (.obs) and column (.var) features
"""


class AnnDataset(AbstractDataset[anndata.AnnData, anndata.AnnData]):
    def __init__(self, filepath: str):
        """Kedro wrapper around anndata.AnnData"""
        # parse the path and protocol (e.g. file, http, s3, etc.) for nodes
        protocol, path = get_protocol_and_path(PurePosixPath(filepath))
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        """Loads data from the .h5ad file.

        Returns:
            Annotated dataframe as AnnData
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        logger.debug(f"Loading anndata from {load_path}!")

        adata = anndata.read_h5ad(load_path)

        logger.debug("Done loading!")
        return adata

    def _save(self, adata: anndata.AnnData) -> None:
        """Saves AnnData to the specified filepath."""

        save_path = get_filepath_str(self._filepath, self._protocol)

        logger.debug(f"Saving anndata nodes to {save_path}")

        adata.write_h5ad(save_path)
        logger.debug(f"Saving anndata edges to {save_path}")

    def _describe(self) -> Dict[str, Any]:
        return {"summy": "dummy"}
