"""Module with GCP datasets for Kedro."""

import os
import re
from copy import deepcopy
from typing import Any, Optional

import google.api_core.exceptions as exceptions
import numpy as np
import pandas as pd
import pygsheets
from google.cloud import bigquery, storage
from kedro.io.core import (
    AbstractVersionedDataset,
    DatasetError,
    Version,
)
from kedro_datasets.spark import SparkDataset, SparkJDBCDataset
from kedro_datasets.spark.spark_dataset import _get_spark, _split_filepath, _strip_dbfs_prefix
from matrix.hooks import SparkHooks
from pygsheets import Spreadsheet, Worksheet
from pyspark.sql import DataFrame, SparkSession
from refit.v1.core.inject import _parse_for_objects


def sanitize_bq_strings(identifier: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(identifier))


class LazySparkDataset(SparkDataset):
    """Lazy loading spark datasets to avoid loading spark every run.

    A trick that makes our spark loading lazy so we never initiate
    """

    def _load(self):
        SparkHooks._initialize_spark()
        return super()._load()


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

    def _load(self) -> DataFrame:
        SparkHooks._initialize_spark()
        load_path = _strip_dbfs_prefix(self._fs_prefix + str(self._get_load_path()))
        read_obj = _get_spark().read

        return read_obj.schema(_parse_for_objects(self._df_schema)).load(
            load_path, self._file_format, **self._load_args
        )


class BigQueryTableDataset(SparkDataset):
    """Dataset to load and save data from BigQuery.

    Kedro dataset to load and write data from BigQuery. This is essentially a wrapper
    for the [BigQuery Spark Connector](https://github.com/GoogleCloudDataproc/spark-bigquery-connector)
    """

    # parquet does not support arrays (our embeddings)
    # https://github.com/GoogleCloudDataproc/spark-bigquery-connector/issues/101
    DEFAULT_SAVE_ARGS = {"intermediateFormat": "orc"}

    def __init__(  # noqa: PLR0913
        self,
        *,
        project_id: str,
        dataset: str,
        table: str,
        identifier: str = None,
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
            dataset: Name of the BigQuery dataset.
            table: name of the table.
            identifier: unique identfier of the table.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
            kwargs: Keyword Args passed to parent.
        """
        self._project_id = project_id
        self._dataset = sanitize_bq_strings(dataset)

        self._table = sanitize_bq_strings("_".join([table, identifier] if identifier else [table]))

        super().__init__(
            filepath="filepath",
            save_args=save_args,
            load_args=load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
            **kwargs,
        )

    def _load(self) -> Any:
        # DEBT potentially a better way would be to globally overwrite the getOrCreate() call in the spark library and link back to SparkSession
        SparkHooks._initialize_spark()
        spark_session = SparkSession.builder.getOrCreate()

        return spark_session.read.format("bigquery").load(f"{self._project_id}.{self._dataset}.{self._table}")

    def _save(self, data: DataFrame) -> None:
        bq_client = bigquery.Client(project=self._project_id)
        dataset_id = f"{self._project_id}.{self._dataset}"

        # Check if the dataset exists
        try:
            bq_client.get_dataset(dataset_id)
            print(f"Dataset {dataset_id} already exists")
        except exceptions.NotFound:
            print(f"Dataset {dataset_id} is not found, will attempt creating it now.")

            # Dataset doesn't exist, so let's create it
            dataset = bigquery.Dataset(dataset_id)
            # dataset.location = "US"  # Specify the location, e.g., "US" or "EU"

            dataset = bq_client.create_dataset(dataset, timeout=30)
            print(f"Created dataset {self._project_id}.{dataset.dataset_id}")

        (
            data.write.format("bigquery")
            .options(**self._save_args)
            .mode("overwrite")
            .save(f"{self._project_id}.{self._dataset}.{self._table}")
        )


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

    def _load(self) -> pd.DataFrame:
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

    def _save(self, data: pd.DataFrame) -> None:
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

    def _load(self) -> Any:
        SparkHooks._initialize_spark()
        bucket = self._get_client().bucket(self._bucket)
        blob = bucket.blob(self._blob_name)

        if not os.path.exists(self._blob_name):
            print("downloading file to local")
            os.makedirs(self._blob_name.rsplit("/", maxsplit=1)[0], exist_ok=True)
            blob.download_to_filename(self._blob_name)
        else:
            print("file present skipping")

        return super()._load()

    def _save(self, df: DataFrame) -> Any:
        raise DatasetError("Save function for RemoteJDBCDataset not implemented!")

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
        fs_prefix, file_name = _split_filepath(file_name)

        return protocol, fs_prefix, file_name
