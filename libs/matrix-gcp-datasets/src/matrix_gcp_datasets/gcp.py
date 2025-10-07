import asyncio
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import fsspec
import google.api_core.exceptions as exceptions
import numpy as np
import pandas as pd
import pygsheets
import pyspark.sql as ps
from google.cloud import bigquery, storage
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetError,
    Version,
)
from kedro_datasets.pandas import GBQTableDataset, ParquetDataset
from kedro_datasets.partitions import PartitionedDataset
from kedro_datasets.spark import SparkDataset, SparkJDBCDataset
from pygsheets import Spreadsheet, Worksheet
from tqdm import tqdm

# TODO: This will need to be injected or made optional when extracting to library
# from matrix.hooks import SparkHooks

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
        provide_empty_if_not_present: bool = False,
    ) -> None:
        self._full_url = filepath
        self._provide_empty_if_not_present = provide_empty_if_not_present

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
        from matrix_gcp_datasets.spark_utils import SparkManager

        SparkManager.initialize_spark()

        # Spark cannot read http files directly
        if self._fs_prefix in ["http://", "https://"]:
            with fsspec.open(self._full_url, "rb") as remote_file:
                name = f"/tmp/{Path(self._full_url).name}"
                with open(name, "wb") as local_file:
                    local_file.write(remote_file.read())
                    self._filepath = Path(name)
                    self._fs_prefix = "file://"

        try:
            return super().load()
        except DatasetError as e:
            if self._provide_empty_if_not_present and ("PATH_NOT_FOUND" in str(e.args)):
                logger.warning(
                    """{"warning": "Dataset not found at '%s'.",  "Resolution": "providing empty dataset with unrelated schema."}""",
                    self._filepath,
                )
                return ps.SparkSession.getActiveSession().createDataFrame(
                    [], schema=ps.types.StructType().add("foo", ps.types.BooleanType())
                )
            else:
                raise e


class SparkBigQueryDataset(LazySparkDataset):
    """
    Spark dataset dat executes a BigQuery query and returns
    the result as a Spark DataFrame.
    """

    def __init__(  # noqa: PLR0913
        self, *, project: str, dataset: str, table: str, shard: str | None = None
    ) -> None:
        super().__init__(
            filepath="dummy", file_format="bigquery", load_args={"table": f"{project}.{dataset}.{table}_{shard}"}
        )

    def _save(self, data: Any) -> None:
        raise NotImplementedError("Save not supported for BigQuerySparkQueryDataSet.")


class PandasBigQueryDataset(GBQTableDataset):
    """
    Pandas dataset that loads data from a BigQuery table and returns
    the result as a Pandas DataFrame.
    """

    def __init__(  # noqa: PLR0913
        self, *, dataset: str, table: str, project: str, shard: str, credentials: dict[str, Any] | None = None
    ) -> None:
        """Creates a new instance of PandasBigQueryDataset.

        Args:
            dataset: BigQuery dataset name.
            table: BigQuery table name.
            project: BigQuery project ID.
            shard: Optional table shard identifier.
            credentials: Optional credentials for BigQuery client.
        """
        super().__init__(dataset=dataset, table_name=f"{table}_{shard}", project=project, credentials=credentials)

    def save(self, data: Any) -> None:
        """Save operation is not supported for this dataset."""
        raise NotImplementedError("Save not supported for PandasBigQueryDataset.")


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
        identifier: str | None = None,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
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
        self._labels = save_args.pop("labels", {}) if save_args else {}

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
        try:
            self._client.create_table(table, exists_ok=False)
        except exceptions.Conflict:
            self._client.update_table(table, fields=["labels"])

    def _create_dataset(self) -> None:
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


"""# NOTE: This class was partially generated using AI assistance."""


class PandasDatasetWithBQExternalTable(ParquetDataset):
    """Pandas dataset that writes Parquet and registers a BQ external table.

    Mirrors ``SparkDatasetWithBQExternalTable`` but uses ``kedro_datasets.pandas.ParquetDataset``
    for IO. After saving, it creates or updates a BigQuery external table pointing to the
    saved location using a wildcard URI.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        project_id: str,
        dataset: str,
        table: str,
        identifier: str | None = None,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._project_id = project_id
        self._path = filepath
        self._format = "parquet"
        self._labels = save_args.pop("labels", {}) if save_args else {}

        self._table = self._sanitize_name("_".join(el for el in [table, identifier] if el is not None))
        self._dataset_id = f"{self._project_id}.{self._sanitize_name(dataset)}"
        self._client = bigquery.Client(project=self._project_id)

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            metadata=metadata,
        )

    def save(self, data: pd.DataFrame) -> None:
        # Invoke saving of the underlying pandas parquet dataset
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
        try:
            self._client.create_table(table, exists_ok=False)
        except exceptions.Conflict:
            self._client.update_table(table, fields=["labels"])

    def _create_dataset(self) -> None:
        try:
            self._client.get_dataset(self._dataset_id)
            logger.info(f"Dataset {self._dataset_id} already exists")
        except exceptions.NotFound:
            logger.info(f"Dataset {self._dataset_id} is not found, will attempt creating it now.")
            dataset = bigquery.Dataset(self._dataset_id)
            dataset = self._client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self._project_id}.{dataset.dataset_id}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
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
        self._sheet: Spreadsheet | None = None

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
        if wks is None and self._sheet:  # type: ignore
            wks = self._sheet.add_worksheet(sheet_name)

        # NOTE: Upon writing, replace empty strings with "" to avoid NaNs in Excel
        data = data.fillna("")

        # Write columns
        for column in self._save_args["write_columns"]:
            col_idx = self._get_col_index(wks, column)

            if col_idx is None:
                raise DatasetError(f"Sheet with {sheet_name} does not contain column {column}!")

            wks.set_dataframe(data[[column]], (1, col_idx + 1)) if wks else None

    @staticmethod
    def _get_wks_by_name(spreadsheet: Spreadsheet, sheet_name: str) -> Worksheet | None:
        for wks in spreadsheet.worksheets():
            if wks.title == sheet_name:
                return wks

        return None

    @staticmethod
    def _get_col_index(sheet: Worksheet, col_name: str) -> int | None:
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

        if fs_prefix and fs_prefix != "gs://":
            raise DatasetError("RemoteSparkJDBCDataset currently supports GCS only")

        if fs_prefix:
            self._bucket, self._blob_name = blob_name.split("/", maxsplit=1)
        else:
            self._bucket = None
            self._blob_name = blob_name

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
        from matrix_gcp_datasets.spark_utils import SparkManager

        SparkManager.initialize_spark()

        if self._bucket and not os.path.exists(self._blob_name):
            logger.info("downloading file to local")
            bucket = self._get_client().bucket(self._bucket)
            blob = bucket.blob(self._blob_name)
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
        timeout: int = 90,
    ) -> None:
        self._max_workers = int(max_workers)
        self._timeout = timeout

        super().__init__(
            path=path,
            dataset=dataset,
            filepath_arg=filepath_arg,
            filename_suffix=filename_suffix,
            credentials=credentials,
            load_args=load_args,
            overwrite=overwrite,
            fs_args=fs_args,
            metadata=metadata,
        )

    def save(self, data: dict[str, Any]) -> None:
        logger.info(f"saving with {self._max_workers} parallelism")

        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)

        # Helper function to process a single partition
        async def process_partition(sem, partition_id, partition_data):
            try:
                # Set up arguments and path
                kwargs = deepcopy(self._dataset_config)
                partition = self._partition_to_path(partition_id)
                kwargs[self._filepath_arg] = self._join_protocol(partition)
                dataset = self._dataset_type(**kwargs)  # type: ignore
                # Evaluate partition data if it's callable
                if callable(partition_data):
                    async with sem:
                        partition_data = await partition_data()  # noqa: PLW2901
                else:
                    raise RuntimeError("not callable")
                # Improve the chances of starting the calculation of a new
                # partition (which is the bottleneck, not the saving), or any
                # other async task, by using cooperative multitasking -> sleep.
                await asyncio.sleep(0.01)

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
                            await asyncio.wait_for(task, self._timeout)
                        except TimeoutError as e:
                            logger.error(
                                f"Timeout error: partition processing took longer than {self._timeout} seconds."
                            )
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
