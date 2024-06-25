"""Script to load maven dependencies."""

from pyspark.sql import SparkSession
from pyspark import SparkConf

import yaml


def main(*args, **kwargs):
    """Main entrypoint.

    Python script to initialize a SparkSession with Kedro's
    configuration for caching purposes.
    """
    # Locate spark configuration
    with open("conf/base/spark.yml") as f:
        parameters = yaml.safe_load(f)

    spark_conf = SparkConf().setAll(parameters.items())

    # Initialise the spark session
    spark_session_conf = SparkSession.builder.appName("tmp").config(conf=spark_conf)
    _spark_session = spark_session_conf.getOrCreate()
    _spark_session.sparkContext.setLogLevel("WARN")


if __name__ == "__main__":
    main()
