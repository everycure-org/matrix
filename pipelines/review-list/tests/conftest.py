from collections.abc import Generator

import pyspark.sql as ps
import pytest


@pytest.fixture(scope="session")
def spark() -> Generator[ps.SparkSession, None, None]:
    """Instantiate the Spark session for testing."""
    spark = (
        ps.SparkSession.builder.config("spark.sql.shuffle.partitions", 1)
        .config("spark.executorEnv.PYTHONPATH", "src")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "1g")
        .master("local")
        .appName("review-list-tests")
        .getOrCreate()
    )
    yield spark
    spark.stop()
