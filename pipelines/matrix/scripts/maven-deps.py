"""Script to load maven dependencies."""

import pyspark.sql as ps
import yaml
from pyspark import SparkConf


def main(*args, **kwargs):
    """Main entrypoint.

    Python script to initialize a SparkSession with Kedro's
    configuration for caching purposes. This script is used as a trick
    when buliding the docker container to bake the `mvn` dependencies
    into the docker image instead of loading them upon container start.
    """
    # Locate spark configuration
    with open("conf/base/spark.yml") as f:
        parameters = yaml.safe_load(f)

    spark_conf = SparkConf().setAll(parameters.items())

    # Initialise the spark session
    spark_session_conf = ps.SparkSession.builder.appName("tmp").config(conf=spark_conf)
    _spark_session = spark_session_conf.getOrCreate()
    _spark_session.sparkContext.setLogLevel("WARN")


if __name__ == "__main__":
    main()
