"""Module containing Neo4JDataset."""

from typing import Any
from copy import deepcopy

from neo4j import GraphDatabase
from pyspark.sql import DataFrame, SparkSession

from kedro.io.core import Version
from kedro_datasets.spark import SparkDataset

import logging

from refit.v1.core.inject import _parse_for_objects

logger = logging.Logger(__name__)


class Neo4JSparkDataset(SparkDataset):
    """Dataset to load and save data from Neo4J.

    Kedro dataset to load and write data from Neo4J. This is essentially a wrapper
    for the [Neo4J spark connector](https://neo4j.com/docs/spark/current/).
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        url: str,
        database: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
        versioned: bool = False,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``.

        Example catalog.yml
        :::

        example.neo4j.nodes:
            type: matrix.datasets.neo4j.Neo4JSparkDataset
            database: database
            url: bolt://127.0.0.1:7687
            credentials:
                authentication.type: basic
                authentication.basic.username: user
                authentication.basic.password: password

            # Map the input dataframe to the graph during writes
            save_args:
                script: >
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:Label) REQUIRE n.property IS UNIQUE;
                query:  >
                    CREATE (n {id: event.id})

            # Map graph to dataframe during read
            load_args:
                # Schema avoids that pyspark needs to run schema inference
                schema:
                    object: pyspark.sql.types.StructType
                    fields:
                        - object: pyspark.sql.types.StructField
                          name: property
                          dataType:
                            object: pyspark.sql.types.StringType
                          nullable: False
                query: >
                    MATCH (n) RETURN n.property as property

        :::

        Args:
            url: URL of the Neo4J instance.
            database: Name of the Neo4J database.
            labels: Labels to filter the nodes.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
            versioned: Flag to decide if we create new databases or stick to the default one.
        """
        self._database = database
        self._url = url
        self._credentials = deepcopy(credentials) or {}

        self._load_args = deepcopy(load_args) or {}
        self._df_schema = self._load_args.pop("schema", None)

        # handle versioned datasets
        # TODO

        super().__init__(
            filepath="filepath",
            save_args=save_args,
            load_args=self._load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
        )

    @staticmethod
    def _create_db(url: str, database: str, credentials: dict[str, Any] = None):
        """Function to create database.

        Args:
            url: URL of the Neo4J instance.
            database: Name of the Neo4J database.
            credentials: neo4j credentials
        """
        # NOTE: Little ugly, as it's extracting out Neo4j spark plugin
        # format. We could pass in user and pass in the dataset, and construct this
        # in options to avoid this.
        creds = (
            credentials.get("authentication.basic.username"),
            credentials.get("authentication.basic.password"),
        )
        with GraphDatabase.driver(url, auth=creds, database="system") as driver:
            # TODO: OR do we want to clear out if exists?
            if database not in Neo4JSparkDataset._load_existing_dbs(driver):
                logging.warning("creating new database %s", database)
                driver.execute_query(f"CREATE DATABASE `{database}` IF NOT EXISTS")

    @staticmethod
    def _load_existing_dbs(driver):
        result = driver.execute_query("SHOW DATABASES")
        dbs = [record["name"] for record in result[0] if record["name"] != "system"]
        return dbs

    def _load(self) -> Any:
        spark_session = SparkSession.builder.getOrCreate()

        load_obj = (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
            .options(**self._load_args)
        )

        if self._df_schema:
            load_obj = load_obj.schema(_parse_for_objects(self._df_schema))

        return load_obj.load()

    def _save(self, data: DataFrame) -> None:
        try:
            if self._save_args.get("persist") == False:
                # skip persistence
                return None
            else:
                # Create database
                self._create_db(self._url, self._database, self._credentials)

                # Write dataset
                (
                    data.write.format("org.neo4j.spark.DataSource")
                    .option("database", self._database)
                    .option("url", self._url)
                    .options(**self._credentials)
                    .options(**self._save_args)
                    .save(**{"mode": "overwrite"})
                )
        except Exception as e:
            logger.error("saving dataset failed with the following parameters")
            logger.error(self._save_args)
            raise e
