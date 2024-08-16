"""Module with GCP datasets for Kedro."""
from typing import Any, Dict, Optional
from copy import deepcopy
import re
from google.cloud import bigquery
import google.api_core.exceptions as exceptions


import pandas as pd

from kedro.io.core import Version
from kedro_datasets.spark import SparkDataset
from kedro_datasets.spark.spark_dataset import _strip_dbfs_prefix, _get_spark
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataset,
    DatasetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

import pygsheets
from pygsheets import Worksheet, Spreadsheet

from pyspark.sql import DataFrame, SparkSession
from matrix.hooks import SparkHooks

from refit.v1.core.inject import _parse_for_objects


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
        identifier: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``.

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
        self._dataset = dataset

        identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", str(identifier))
        self._table = f"{table}_{identifier}"

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

        return spark_session.read.format("bigquery").load(
            f"{self._project_id}.{self._dataset}.{self._table}"
        )

    def _save(self, data: DataFrame) -> None:
        bq_client = bigquery.Client()
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

        return df

    def _save(self, data: pd.DataFrame) -> None:
        self._init_sheet()

        sheet_name = self._save_args["sheet_name"]
        wks = self._get_wks_by_name(self._sheet, sheet_name)

        # Create the worksheet if not exists
        if wks is None:
            wks = self._sheet.add_worksheet(sheet_name)

        # Write columns
        for column in self._save_args["write_columns"]:
            col_idx = self._get_col_index(wks, column)

            if col_idx is None:
                raise DatasetError(
                    f"Sheet with {sheet_name} does not contain column {column}!"
                )

            wks.set_dataframe(data[[column]], (1, col_idx + 1))

    @staticmethod
    def _get_wks_by_name(
        spreadsheet: Spreadsheet, sheet_name: str
    ) -> Optional[Worksheet]:
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
