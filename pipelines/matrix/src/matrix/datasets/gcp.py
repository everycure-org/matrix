import asyncio
import logging
import os
import re
from copy import deepcopy
from typing import Any, Optional

import google.api_core.exceptions as exceptions
import numpy as np
import pandas as pd
import pygsheets
import pyspark.sql as ps
import requests
from google.cloud import bigquery, storage
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetError,
    Version,
)
from kedro_datasets.partitions import PartitionedDataset
from kedro_datasets.spark import SparkDataset, SparkJDBCDataset
from matrix.hooks import SparkHooks
from matrix.inject import _parse_for_objects
from pygsheets import Spreadsheet, Worksheet
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LazySparkDataset(SparkDataset):
    """Lazy loading spark datasets to avoid loading spark every run.

    A trick that makes our spark loading lazy so we never initiate
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        file_format: str = "parquet",
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = filepath

        super().__init__(
            filepath=filepath,
            file_format=file_format,
            save_args=save_args,
            load_args=load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
        )

    def load(self):
        SparkHooks._initialize_spark()

        # Make local copy
        if self.self._filepath.startswith("https://"):
            breakpoint()

        return super().load()


class SparkWithSchemaDataset(SparkDataset):
    """Dataset to load BigQuery data.

    Dataset extends the behaviour of the standard SparkDataset with
    a schema load argument that can be specified in the Data catalog.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        file_format: str = "parquet",
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of ``SparkWithSchemaDataset``."""
        self._load_args = deepcopy(load_args) or {}
        self._df_schema = self._load_args.pop("schema")

        super().__init__(
            filepath=filepath,
            file_format=file_format,
            save_args=save_args,
            load_args=self._load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
        )

    def load(self) -> ps.DataFrame:
        SparkHooks._initialize_spark()
        load_path = self._strip_dbfs_prefix(self._fs_prefix + str(self._get_load_path()))
        read_obj = ps.SparkSession.builder.getOrCreate()

        return read_obj.schema(_parse_for_objects(self._df_schema)).load(
            load_path, self._file_format, **self._load_args
        )

    @staticmethod
    def _strip_dbfs_prefix(path: str, prefix: str = "/dbfs") -> str:
        return path[len(prefix) :] if path.startswith(prefix) else path


class SparkDatasetWithBQExternalTable(LazySparkDataset):
    """Spark Dataset that produces a BigQuery external table as a side output.

    The class delegates dataset save and load invocations to the native SparkDataset, and registers the
    dataset into BigQuery through External Data Configuration. This means that the dataset is visible in BQ, but we do not incur unnecessary costs for BQ IO.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        project_id: str,
        dataset: str,
        table: str,
        file_format: str,
        identifier: Optional[str] = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Creates a new instance of ``BigQueryTableDataset``.

        Args:
            project_id: project identifier.
            filepath: filepath to write to
            dataset: Name of the BigQuery dataset.
            table: name of the table.
            identifier: unique identfier of the table.
            file_format: file format to use
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
            kwargs: Keyword Args passed to parent.
        """
        self._project_id = project_id
        self._path = filepath
        self._format = file_format
        self._labels = save_args.pop("labels", {})

        self._table = self._sanitize_name("_".join(el for el in [table, identifier] if el is not None))
        self._dataset_id = f"{self._project_id}.{self._sanitize_name(dataset)}"

        self._client = bigquery.Client(project=self._project_id)

        super().__init__(
            filepath=filepath,
            file_format=file_format,
            save_args=save_args,
            load_args=load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
            **kwargs,
        )

    def save(self, data: ps.DataFrame) -> None:
        # Invoke saving of the underlying spark dataset
        super().save(data)

        # Ensure dataset exists
        self._create_dataset()

        # Create external table, referencing the dataset in object storage
        external_config = bigquery.ExternalConfig(self._format.upper())
        external_config.source_uris = [f"{self._path}/*.{self._format}"]

        # Register the external table within BigQuery
        table = bigquery.Table(f"{self._dataset_id}.{self._table}")
        table.labels = self._labels
        table.external_data_configuration = external_config
        table = self._client.create_table(table, exists_ok=True)

    def _create_dataset(self) -> str:
        try:
            self._client.get_dataset(self._dataset_id)
            logger.info(f"Dataset {self._dataset_id} already exists")
        except exceptions.NotFound:
            logger.info(f"Dataset {self._dataset_id} is not found, will attempt creating it now.")

            # Dataset doesn't exist, so let's create it
            dataset = bigquery.Dataset(self._dataset_id)

            dataset = self._client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self._project_id}.{dataset.dataset_id}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Function to sanitize BigQuery table or dataset identifiers.

        Args:
            name: str
        Returns:
            Sanitized name
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))


class GoogleSheetsDataset(AbstractVersionedDataset[pd.DataFrame, pd.DataFrame]):
    """Dataset to load data from Google sheets."""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str,
        service_file: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of ``GoogleSheetsDataset``.

        Args:
            key: Google sheets key
            service_file: path to service accunt file.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
            kwargs: Keyword Args passed to parent.
        """
        self._key = key
        self._service_file = service_file
        self._sheet = None

        super().__init__(
            filepath=None,
            version=version,
            exists_function=self._exists,
            glob_function=None,
        )

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _init_sheet(self):
        """Function to initialize the spreadsheet.

        This is executed lazily to avoid loading credentials on python runtime launch which creates issues
        in unit tests.
        """
        if self._sheet is None:
            gc = pygsheets.authorize(service_file=self._service_file)
            self._sheet = gc.open_by_key(self._key)

    def load(self) -> pd.DataFrame:
        self._init_sheet()

        sheet_name = self._load_args["sheet_name"]
        wks = self._get_wks_by_name(self._sheet, sheet_name)
        if wks is None:
            raise DatasetError(f"Sheet with name {sheet_name} not found!")

        df = wks.get_as_df()
        if (cols := self._load_args.get("columns", None)) is not None:
            df = df[cols]

        # NOTE: Upon loading, replace empty strings with NaN
        return df.replace(r"^\s*$|^N/A$", np.nan, regex=True)

    def save(self, data: pd.DataFrame) -> None:
        self._init_sheet()

        sheet_name = self._save_args["sheet_name"]
        wks = self._get_wks_by_name(self._sheet, sheet_name)

        # Create the worksheet if not exists
        if wks is None:
            wks = self._sheet.add_worksheet(sheet_name)

        # NOTE: Upon writing, replace empty strings with "" to avoid NaNs in Excel
        data = data.fillna("")

        # Write columns
        for column in self._save_args["write_columns"]:
            col_idx = self._get_col_index(wks, column)

            if col_idx is None:
                raise DatasetError(f"Sheet with {sheet_name} does not contain column {column}!")

            wks.set_dataframe(data[[column]], (1, col_idx + 1))

    @staticmethod
    def _get_wks_by_name(spreadsheet: Spreadsheet, sheet_name: str) -> Optional[Worksheet]:
        for wks in spreadsheet.worksheets():
            if wks.title == sheet_name:
                return wks

        return None

    @staticmethod
    def _get_col_index(sheet: Worksheet, col_name: str) -> Optional[int]:
        for idx, col in enumerate(sheet.get_row(1)):
            if col == col_name:
                return idx

        return None

    def _describe(self) -> dict[str, Any]:
        return {
            "key": self._key,
        }

    def _exists(self) -> bool:
        return False


class RemoteSparkJDBCDataset(SparkJDBCDataset):
    """Dataset to allow connection to remote JDBC dataset.

    The JDBC dataset is restricted in the sense that it only allows urls from local. This
    dataset provides an adaptor to reference to datasets in google cloud storage. The dataset works
    by downloading the file into local, providing a local cache.

    NOTE: Only works for datasets in Google Cloud Storage
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        project: str,
        table: str,
        url: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of ``RemoteSparkJDBCDataset``."""
        self._client = None
        self._project = project

        protocol, fs_prefix, blob_name = self.split_remote_jdbc_path(url)

        if fs_prefix != "gs://":
            raise DatasetError("RemoteSparkJDBCDataset currently supports GCS only")

        self._bucket, self._blob_name = blob_name.split("/", maxsplit=1)

        super().__init__(
            table=table,
            url=f"jdbc:{protocol}:{self._blob_name}",
            load_args=load_args,
            save_args=save_args,
            credentials=credentials,
            metadata=metadata,
        )

    def _get_client(self):
        """Lazily initialize the GCS client when needed.

        NOTE: This is a workaround to avoid the GCS client being initialized on every run.
        as it would require an authenticated environment even for unit tests.
        """
        if self._client is None:
            self._client = storage.Client(self._project)
        return self._client

    def load(self) -> Any:
        SparkHooks._initialize_spark()
        bucket = self._get_client().bucket(self._bucket)
        blob = bucket.blob(self._blob_name)

        if not os.path.exists(self._blob_name):
            logger.info("downloading file to local")
            os.makedirs(self._blob_name.rsplit("/", maxsplit=1)[0], exist_ok=True)
            blob.download_to_filename(self._blob_name)
        else:
            logger.info("file present skipping")

        return super().load()

    def save(self, df: ps.DataFrame) -> Any:
        super().save(df)

    @staticmethod
    def split_remote_jdbc_path(url: str):
        """Function to split jdbc path into components.

        Args:
            url: URL
            fs_prefix: filesystem prefix
        Returns:
            protocol: jdbc protocol
            fs_prefix: filesystem prefix
        """
        protocol, file_name = url.split(":", maxsplit=1)
        fs_prefix, file_name = RemoteSparkJDBCDataset._split_filepath(file_name)

        return protocol, fs_prefix, file_name

    @staticmethod
    def _split_filepath(filepath: str | os.PathLike) -> tuple[str, str]:
        split_ = str(filepath).split("://", 1)
        if len(split_) == 2:  # noqa: PLR2004
            return split_[0] + "://", split_[1]
        return "", split_[0]


class PartitionedAsyncParallelDataset(PartitionedDataset):
    """
    Custom implementation of the ParallelDataset that allows concurrent processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        path: str,
        dataset: str | type[AbstractDataset] | dict[str, Any],
        max_workers: int = 10,
        filepath_arg: str = "filepath",
        filename_suffix: str = "",
        credentials: dict[str, Any] | None = None,
        load_args: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._max_workers = int(max_workers)

        super().__init__(
            path=path,
            dataset=dataset,
            filepath_arg=filepath_arg,
            filename_suffix=filename_suffix,
            credentials=credentials,
            load_args=load_args,
            overwrite=overwrite,
            fs_args=fs_args,
        )

    def save(self, data: dict[str, Any], timeout: int = 60) -> None:
        logger.info(f"saving with {self._max_workers} parallelism")

        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)

        # Helper function to process a single partition
        async def process_partition(sem, partition_id, partition_data):
            async with sem:
                try:
                    # Set up arguments and path
                    kwargs = deepcopy(self._dataset_config)
                    partition = self._partition_to_path(partition_id)
                    kwargs[self._filepath_arg] = self._join_protocol(partition)
                    dataset = self._dataset_type(**kwargs)  # type: ignore

                    # Evaluate partition data if it's callable
                    if callable(partition_data):
                        partition_data = await partition_data()  # noqa: PLW2901
                    else:
                        raise RuntimeError("not callable")

                    # Save the partition data
                    dataset.save(partition_data)
                except Exception as e:
                    logger.error(f"Error in process_partition with partition {partition_id}: {e}")
                    raise

        # Define function to run asyncio tasks within a synchronous function
        def run_async_tasks():
            # Create an event loop and a thread pool executor for async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sem = asyncio.Semaphore(self._max_workers)

            tasks = [
                loop.create_task(process_partition(sem, partition_id, partition_data))
                for partition_id, partition_data in sorted(data.items())
            ]

            # Track progress with tqdm as tasks complete
            with tqdm(total=len(tasks), desc="Saving partitions") as progress_bar:

                async def monitor_tasks():
                    for task in asyncio.as_completed(tasks):
                        try:
                            await asyncio.wait_for(task, timeout)
                        except asyncio.TimeoutError as e:
                            logger.error(f"Timeout error: partition processing took longer than {timeout} seconds.")
                            raise e
                        except Exception as e:
                            logger.error(f"Error processing partition in tqdm loop: {e}")
                            raise e
                        finally:
                            progress_bar.update(1)

                # Run the monitoring coroutine
                try:
                    loop.run_until_complete(monitor_tasks())
                finally:
                    loop.close()

        run_async_tasks()
        self._invalidate_caches()
