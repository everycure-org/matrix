"""Custom Kedro datasets for RDF graph processing.

This module provides dataset classes for loading and saving RDF/OWL ontologies:
- PyOxiGraphDataset: Uses PyOxigraph Store for fast SPARQL querying
"""

import logging
from pathlib import Path
from typing import Any

import pyoxigraph
import requests
from kedro.io.core import AbstractDataset, DatasetError
from pyoxigraph import RdfFormat

logger = logging.getLogger(__name__)


class PyOxiGraphDataset(AbstractDataset):
    """Kedro dataset for loading and saving PyOxigraph Store objects.

    This dataset handles RDF/OWL ontology files and provides them as
    PyOxigraph Store objects for fast in-memory SPARQL querying and manipulation.

    Example usage in catalog:
        mondo_graph:
            type: matrix.datasets.rdf.PyOxiGraphDataset
            filepath: data/01_raw/mondo.owl
            load_args:
                format: xml  # or ttl, n3, nt, etc.
            save_args:
                format: xml
    """

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the PyOxiGraphDataset.

        Args:
            filepath: Path to the RDF/OWL file (local path or URL)
            load_args: Additional arguments for PyOxigraph Store.load()
                - format: RDF format (xml, ttl, n3, nt, etc.). Default: "xml"
            save_args: Additional arguments for PyOxigraph Store.dump()
                - format: Output format. Default: "xml"
            metadata: Any arbitrary metadata (used by Kedro)
        """
        self._filepath = filepath  # Keep as string to support URLs
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._metadata = metadata or {}

        # Set default format to XML (for OWL files) if not specified
        if "format" not in self._load_args:
            self._load_args["format"] = "xml"
        if "format" not in self._save_args:
            self._save_args["format"] = "xml"

    def _load(self):
        """Load RDF/OWL file into a PyOxigraph Store for fast SPARQL queries.

        Returns:
            PyOxigraph Store object with parsed ontology

        Raises:
            DatasetError: If file cannot be loaded or parsed
        """
        try:
            store = pyoxigraph.Store()

            # Map format string to PyOxigraph RdfFormat
            format_map = {
                "xml": RdfFormat.RDF_XML,
                "ttl": RdfFormat.TURTLE,
                "nt": RdfFormat.N_TRIPLES,
                "n3": RdfFormat.N3,
            }
            format_str = self._load_args.get("format", "xml")
            rdf_format = format_map.get(format_str, RdfFormat.RDF_XML)

            # Load from URL or file
            if self._filepath.startswith(("http://", "https://")):
                # Download from URL
                response = requests.get(self._filepath)
                response.raise_for_status()
                content = response.content
                store.load(content, format=rdf_format)
            else:
                # Load from file using path parameter (more efficient)
                store.load(path=self._filepath, format=rdf_format)
            return store
        except Exception as e:
            raise DatasetError(
                f"Failed to load RDF graph from {self._filepath}: {e}"
            ) from e

    def _save(self, store) -> None:
        """Save PyOxigraph Store to file.

        Args:
            store: PyOxigraph Store object to save

        Raises:
            DatasetError: If store cannot be saved
        """
        try:

            # Ensure parent directory exists for local files
            filepath_obj = Path(self._filepath)
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)

            # PyOxigraph dump() requires dataset formats (N-Quads or TriG)
            # We always use N-Quads for serialization regardless of requested format
            

            logger.info("Saving PyOxigraph store as N-Quads format")
            serialized = store.dump(format=RdfFormat.N_QUADS)

            with open(self._filepath, 'wb') as f:
                f.write(serialized)

        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(
                f"Failed to save RDF graph: {e}"
            ) from e

    def _describe(self) -> dict[str, Any]:
        """Return a description of the dataset.

        Returns:
            Dictionary describing the dataset configuration
        """
        return {
            "filepath": str(self._filepath),
            "load_args": self._load_args,
            "save_args": self._save_args,
        }
