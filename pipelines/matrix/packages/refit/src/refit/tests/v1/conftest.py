# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
# pylint: skip-file
# flake8: noqa
"""Contains fixtures."""

import datetime
import time

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


@pytest.fixture(scope="module")
def spark():
    """Prepare a spark session."""
    spark = SparkSession.builder.config("spark.sql.shuffle.partitions", 1).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def dummy_spark_df(spark):
    """Dummy spark dataframe."""
    dummy_pd_df = pd.DataFrame([{"c1": 1}])
    dummy_spark_df = spark.createDataFrame(dummy_pd_df)
    return dummy_spark_df


@pytest.fixture
def dummy_pd_df():
    """Dummy pandas dataframe."""
    dummy_pd_df = pd.DataFrame([{"c1": 1}])
    return dummy_pd_df


def dummy_func(x):
    """Dummy function for testing purposes."""
    return x


def dummy_func_without_input():
    """Dummy function for testing purposes."""
    return "hello world"


class CustomException(Exception):
    """Custom exception class for testing purposes."""

    pass


# ----- has_schema pyspark -----
@pytest.fixture
def sample_df_spark_all_dtypes(spark):
    """Sample spark dataframe with all dtypes."""
    timestamp = datetime.datetime.fromtimestamp(time.time())

    schema = StructType(
        [
            StructField("int_col", IntegerType(), True),
            StructField("long_col", LongType(), True),
            StructField("string_col", StringType(), True),
            StructField("float_col", FloatType(), True),
            StructField("double_col", DoubleType(), True),
            StructField("date_col", DateType(), True),
            StructField("datetime_col", TimestampType(), True),
            StructField("array_int", ArrayType(IntegerType()), True),
        ]
    )

    data = [
        (
            1,
            2,
            "awesome string",
            10.01,
            0.89,
            pd.Timestamp("2012-05-01").date(),
            timestamp,
            [1, 2, 3],
        ),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_df_pd_all_dtypes():
    """Sample pandas dataframe with all dtypes."""
    df = pd.DataFrame(
        {
            "float_col": [1.0],
            "int_col": [1],
            "datetime_col": [pd.Timestamp("20180310")],
            "date_col": [pd.Timestamp("20180310").date()],
            "string_col": ["foo"],
        }
    )

    df["datetime_ms_col"] = df["datetime_col"].values.astype("datetime64[ms]")

    return df


@pytest.fixture
def spark_df(spark):
    """Sample spark dataframe with all dtypes."""
    schema = StructType(
        [
            StructField("int_col", IntegerType(), True),
            StructField("float_col", FloatType(), True),
            StructField("string_col", StringType(), True),
        ]
    )

    data = [
        (1, 2.0, "awesome string",),
        (2, 2.0, None,),
        (3, 2.0, "hello world",),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def pandas_df():
    """Sample pandas dataframe with all dtypes."""
    data = [
        {"float_col": 1.0, "int_col": 1, "string_col": "foo",},
        {"float_col": 1.0, "int_col": 2, "string_col": "blabla",},
        {"float_col": 1.0, "int_col": 3, "string_col": None,},
    ]
    df = pd.DataFrame(data)

    return df


@pytest.fixture
def spark_nulls_df(spark):
    """Sample spark dataframe with all dtypes."""
    schema = StructType(
        [
            StructField("int_col", IntegerType(), True),
            StructField("float_col", FloatType(), True),
            StructField("string_col", StringType(), True),
            StructField("bool_col", BooleanType(), True),
        ]
    )

    data = [
        (1, None, "awesome string", None),
        (2, 2.0, None, False),
        (3, 2.0, "hello world", None),
        (None, None, "random sample", False),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def pandas_nulls_df():
    """Sample pandas dataframe with all dtypes."""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, None],
            "float_col": [None, 2.0, 2.0, None],
            "string_col": ["foo", None, "sample", "awesome"],
            "bool_col": [True, None, None, True],
        }
    )

    return df


@pytest.fixture
def spark_df_with_pk(spark):
    """Sample spark dataframe with primary key column."""
    schema = StructType(
        [StructField("id", IntegerType()), StructField("name", StringType())]
    )

    data = [(1, "Adam"), (2, "Smith"), (None, "George")]

    return spark.createDataFrame(data, schema)


@pytest.fixture()
def pandas_df_with_pk():
    """Sample pandas dataframe with primary key column."""
    data = [
        {"id1": 101, "id2": "A_01", "name": "Adam"},
        {"id1": 102, "id2": "A_02", "name": "Smith"},
        {"id1": 102, "id2": "A_02", "name": "George"},
        {"id1": 104, "id2": "A_04", "name": "Jane"},
        {"id1": 105, "id2": None, "name": "John"},
    ]

    return pd.DataFrame(data)


@pytest.fixture()
def pandas_pk_null_df():
    """Sample pandas dataframe with nulls in primary key column."""
    data = [
        {"id1": 101, "id2": None,},
        {"id1": 102, "id2": "A_02",},
        {"id1": 103, "id2": "A_03",},
        {"id1": None, "id2": None,},
        {"id1": 104, "id2": None,},
    ]

    return pd.DataFrame(data)


@pytest.fixture()
def pandas_string_df():
    """Sample pandas dataframe with string dtypes."""
    return pd.DataFrame({"string_col": ["Asia", "Africa", "Europe"],})
