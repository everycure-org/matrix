from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class Parser(ABC):
    """Abstract base class for PKS metadata parsers.
    Example:
        parser:
          _object: matrix.pipelines.document_kg.parsers.InforesParser
          name: infores
          id_column: id
          extracted_metadata: [id, name, description]
    """

    def __init__(self, name: str, id_column: str, extracted_metadata: list[str], **config_params):
        """Initialize parser with configuration.

        Args:
            name: Parser name (e.g., 'infores', 'reusabledata')
            id_column: Column name containing the primary identifier
            extracted_metadata: List of metadata fields to extract
            **config_params: Additional configuration parameters
        """
        self.name = name
        self.id_column = id_column
        self.extracted_metadata = extracted_metadata
        self.config = config_params

    @abstractmethod
    def parse(
        self, source_data: Any, mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Parse source data into standardized PKS metadata."""
        raise NotImplementedError


class ExternalRegistryParser(Parser):
    """Base class for external registry parsers (infores, reusabledata, kgregistry)."""

    pass


class MatrixCuratedParser(Parser):
    """Base class for Matrix-curated parsers (internal curation and reviews)."""

    pass
