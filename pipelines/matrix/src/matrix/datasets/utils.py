"""Module containing utility functions for the datasets."""

from typing import Any, Dict

import logging

from kedro_datasets.pandas import ParquetDataset
from kedro.io.core import DatasetError, Version

logger = logging.getLogger(__name__)


class BaseParquetDataset(ParquetDataset):
    """Base class for Parquet datasets with retry logic."""

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
        """Initializes the BaseParquetDataset."""
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

    def _load_with_retry(self, result_class, result_class_arg=None, num_retries=3):
        """Load the dataset with retry logic.

        Args:
            result_class: The class to load the result as.
            result_class_arg: Name of the argument to pass to the result class constructor.
        """
        attempt = 0

        # Retrying due to a very flaky error, causing GCS not retrieving
        # parquet files on first try.
        while attempt < num_retries:
            try:
                # Attempt reading the object
                # https://github.com/everycure-org/matrix/issues/7
                if result_class_arg is not None:
                    return result_class(**{result_class_arg: super()._load()})
                else:
                    return result_class(super()._load())
            except FileNotFoundError:
                attempt += 1
                logger.warning("Parquet file not found, retrying!")

        raise DatasetError("Unable to find underlying Parquet file!")
