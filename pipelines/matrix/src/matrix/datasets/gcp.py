"""Module with GCP datasets for Kedro."""
from copy import deepcopy
from typing import Any, Dict, Union

from kedro.io.core import Version, validate_on_forbidden_chars
from kedro_datasets.pandas import GBQTableDataset
from kedro_datasets.spark import SparkDataset
from kedro_datasets.spark.spark_dataset import _strip_dbfs_prefix, _get_spark
from google.cloud import bigquery
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

    def __init__(  # noqa: PLR0913
        self,
        *,
        project_id: str,
        dataset: str,
        table: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``.

        Args:
            project_id: project identifier.
            dataset: Name of the BigQuery dataset.
            table: name of the table.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
        """
        self._project_id = project_id
        self._dataset = dataset
        self._table = table

        super().__init__(
            filepath="filepath",
            save_args=save_args,
            load_args=load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
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


class MatrixGBQQueryDataset(GBQTableDataset):

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"progress_bar": False}

    def __init__(  # noqa: PLR0913
            self,
            *,
            dataset: str,
            table_name: str,
            project: str = None,
            credentials: dict[str, Any] | None = None,
            load_args: dict[str, Any] = None,
            save_args: dict[str, Any] = None,
            metadata: dict[str, Any] = None,
        ) -> None:
            """Creates a new instance of ``GBQTableDataset``.

            Args:
                dataset: Google BigQuery dataset.
                table_name: Google BigQuery table name.
                project: Google BigQuery Account project ID.
                    Optional when available from the environment.
                    https://cloud.google.com/resource-manager/docs/creating-managing-projects
                credentials: Credentials for accessing Google APIs.
                    Either ``google.auth.credentials.Credentials`` object or dictionary with
                    parameters required to instantiate ``google.oauth2.credentials.Credentials``.
                    Here you can find all the arguments:
                    https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.credentials.html
                load_args: Pandas options for loading BigQuery table into DataFrame.
                    Here you can find all available arguments:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_gbq.html
                    All defaults are preserved.
                save_args: Pandas options for saving DataFrame to BigQuery table.
                    Here you can find all available arguments:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_gbq.html
                    All defaults are preserved, but "progress_bar", which is set to False.
                metadata: Any arbitrary metadata.
                    This is ignored by Kedro, but may be consumed by users or external plugins.

            Raises:
                DatasetError: When ``load_args['location']`` and ``save_args['location']``
                    are different.
            """
            # Handle default load and save arguments
            self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
            if load_args is not None:
                self._load_args.update(load_args)
            self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
            if save_args is not None:
                self._save_args.update(save_args)

            self._validate_location()
            validate_on_forbidden_chars(dataset=dataset, table_name=table_name)
            self._client = bigquery.Client.from_service_account_info(credentials)

            # if isinstance(credentials, dict):
            #     credentials = Credentials(**credentials)

            self._dataset = dataset
            self._table_name = table_name
            self._project_id = project
            self._credentials = credentials
            # self._client = bigquery.Client(
            #     project=self._project_id,
            #     credentials=self._credentials,
            #     location=self._save_args.get("location"),
            # )

            self.metadata = metadata

