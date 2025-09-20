import asyncio
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import fsspec
import google.api_core.exceptions as exceptions
import gspread
import pandas as pd
import pyspark.sql as ps
from google.cloud import bigquery, storage
from kedro.io.core import (
    AbstractDataset,
    DatasetError,
    Version,
)
from kedro_datasets.pandas import CSVDataset, GBQTableDataset
from kedro_datasets.partitions import PartitionedDataset
from kedro_datasets.spark import SparkDataset, SparkJDBCDataset
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


class GoogleSheetsDataset(CSVDataset):
    """Dataset to load and save data from Google sheets."""

    def __init__(
        self,
        spreadsheet_url: str,
        worksheet_gid: str,
        service_account_file_path: str,
    ) -> None:
        """
        Args:
            spreadsheet_url: URL of a spreadsheet as it appears in a browser.
            worksheet_gid: The id of a worksheet. it can be seen in the url as the value of the parameter ‘gid’
            service_account_file_path: Path to the service account file. The Google Sheet must be shared with this service account's email.
        """
        self._gc = gspread.service_account(filename=service_account_file_path)
        self._worksheet = self._gc.open_by_url(spreadsheet_url).get_worksheet_by_id(worksheet_gid)

        super().__init__(filepath=None)

    def load(self) -> pd.DataFrame:
        df = pd.DataFrame(self._worksheet.get_all_records())
        return df

    def save(self, df: pd.DataFrame) -> None:
        self._worksheet.update([df.columns.values.tolist()] + df.values.tolist())


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
