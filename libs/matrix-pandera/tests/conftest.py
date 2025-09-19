from typing import Generator

import pyspark.sql as ps
import pytest


@pytest.fixture(scope="session")
def spark() -> Generator[ps.SparkSession, None, None]:
    """Instantiate the Spark session."""
    spark = (
        ps.SparkSession.builder.config("spark.sql.shuffle.partitions", 1)
        .config("spark.executorEnv.PYTHONPATH", "src")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.showConsoleProgress", "false")
        .master("local")
        .appName("tests")
        .getOrCreate()
    )
    yield spark
