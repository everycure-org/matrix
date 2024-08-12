"""Module with GCP datasets for Kedro."""
from typing import Any, Dict
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
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of ``GoogleSheetsDataset``.

        Args:
            key: Google sheets key
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
            kwargs: Keyword Args passed to parent.
        """
        self._key = key
        self._sheet_name = load_args.pop("sheet_name")
        self._gc = pygsheets.authorize(service_file="conf/local/service-account.json")

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> pd.DataFrame:
        sheet = self._gc.open_by_key(self._key)

        for wks in sheet.worksheets():
            if wks.title == self._sheet_name:
                return wks.get_as_df()

        raise DatasetError(f"Sheet with name {self._sheet_name} not found!")

    def _save(self, data: pd.DataFrame) -> None:
        raise NotImplementedError("Save method not yet implemented!")

    def _describe(self) -> dict[str, Any]:
        return {
            "key": self._key,
        }
