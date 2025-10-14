# NOTE: This file was partially generated using AI assistance.

"""Matrix datasets module.

This module contains custom Kedro datasets for the MATRIX pipeline,
including integrations with various data sources and formats.
"""

from .huggingface import (
    HuggingFaceBaseDataset,
    HuggingFaceParquetDataset,
    HuggingFaceCSVDataset,
    HuggingFaceJSONDataset,
    HuggingFaceXetDataset,
    HuggingFaceDataset,  # Alias for HuggingFaceParquetDataset
    HuggingFaceDatasetError,
    AuthenticationError,
    HuggingFaceRepositoryNotFoundError,
    HuggingFaceFileNotFoundError,
)

__all__ = [
    # HuggingFace datasets
    "HuggingFaceBaseDataset",
    "HuggingFaceParquetDataset", 
    "HuggingFaceCSVDataset",
    "HuggingFaceJSONDataset",
    "HuggingFaceXetDataset",
    "HuggingFaceDataset",
    # Exceptions
    "HuggingFaceDatasetError",
    "AuthenticationError", 
    "HuggingFaceRepositoryNotFoundError",
    "HuggingFaceFileNotFoundError",
]