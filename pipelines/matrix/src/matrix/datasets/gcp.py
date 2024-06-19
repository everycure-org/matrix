"""Module with GCP datasets for Kedro."""
from typing import Any, Dict

from kedro.io.core import Version
from kedro_datasets.spark import SparkDataset

from pyspark.sql import DataFrame, SparkSession


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
        labels: Dict[str, str] = None,
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
            labels: kv-pairs added as labels to table.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
        """
        self._project_id = project_id
        self._dataset = dataset
        self._table = table
        self._labels = labels

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
        write_obj = (
            data.write.format("bigquery")
            .option(
                "writeMethod",
                "direct",
            )  # NOTE: Alternatively, we can define tmp. gcp location
            .mode("overwrite")
        )

        if self._labels:
            for label, value in self._labels.items():
                write_obj = write_obj.option(f"bigQueryTableLabel.{label}", value)

        write_obj.save(f"{self._project_id}.{self._dataset}.{self._table}")
