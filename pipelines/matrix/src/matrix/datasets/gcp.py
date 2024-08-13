"""Module with GCP datasets for Kedro."""
from typing import Any, Dict, Optional
from copy import deepcopy
import re

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

from refit.v1.core.inject import _parse_for_objects


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

        identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
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
        spark_session = SparkSession.builder.getOrCreate()

        return spark_session.read.format("bigquery").load(
            f"{self._project_id}.{self._dataset}.{self._table}"
        )

    def _save(self, data: DataFrame) -> None:
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
        gc = pygsheets.authorize(service_file=service_file)
        self._key = key
        self._sheet = gc.open_by_key(self._key)

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

    def _load(self) -> pd.DataFrame:
        wks = self._get_wks_by_name(self._sheet, self._load_args["sheet_name"])
        if wks is None:
            raise DatasetError(f"Sheet with name {self._sheet_name} not found!")

        return wks.get_as_df()

    def _save(self, data: pd.DataFrame) -> None:
        wks = self._get_wks_by_name(self._sheet, self._save_args["sheet_name"])

        # Create the worksheet if not exists
        if wks is None:
            wks = self._sheet.add_worksheet(self._save_args["sheet_name"])

        for column in self._save_args["write_columns"]:
            col_idx = self._get_col_index(wks, column)

            if col_idx is None:
                raise DatasetError(
                    f"Sheet with {self._save_args['sheet_name']} does not contain column {column}!"
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
