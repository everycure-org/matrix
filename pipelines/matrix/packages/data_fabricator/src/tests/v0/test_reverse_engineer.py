# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided 'as is', without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey's use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
"""Tests."""

# pylint: skip-file
# flake8: noqa
import datetime

import pandas as pd
import pytest
import yaml
import pyspark.sql as ps
from pyspark.sql.types import (
    ArrayType,
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

from data_fabricator.v0.core.fabricator import MockDataGenerator
from data_fabricator.v0.core.reverse_engineer import (
    reverse_engineer_df,
    reverse_engineer_tables,
)


@pytest.fixture
def spark():
    return ps.SparkSession.builder.getOrCreate()


@pytest.fixture
def sample_df(spark):
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
            datetime.datetime(2020, 11, 30, 18, 29, 19, 990601),
            [1, 2, 3],
        ),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def expected_str():
    expected_str = """
        columns:
          date_col:
            sample_values:
            - 2012-05-01
            type: generate_values
          datetime_col:
            sample_values:
            - 2020-11-30 18:29:19.990601
            type: generate_values
          double_col:
            sample_values:
            - 0.89
            type: generate_values
          float_col:
            sample_values:
            - 10.010000228881836
            type: generate_values
          int_col:
            sample_values:
            - 1
            type: generate_values
          long_col:
            sample_values:
            - 2
            type: generate_values
          string_col:
            sample_values:
            - awesome string
            type: generate_values
        num_rows: 10
    """
    return expected_str


def test_reverse_engineer_spark(sample_df, expected_str):
    valid_types_only = sample_df.drop("array_int")

    valid_types_replicated = valid_types_only.union(valid_types_only).union(
        valid_types_only
    )

    table_config = reverse_engineer_df(df=valid_types_replicated, num_rows=10)

    config_str = yaml.dump(data=table_config)

    assert yaml.safe_load(config_str) == yaml.safe_load(expected_str)


def test_reverse_engineer_pandas(sample_df):
    valid_types_only = sample_df.drop("array_int")

    valid_types_replicated = (
        valid_types_only.union(valid_types_only).union(valid_types_only).toPandas()
    )

    with pytest.raises(NotImplementedError):
        reverse_engineer_df(df=valid_types_replicated, num_rows=10)


def test_bad_spark_dtypes(sample_df):
    with pytest.raises(ValueError):
        reverse_engineer_df(df=sample_df, num_rows=10)


@pytest.fixture
def expected_config_str_tables():
    expected_config_str_tables = """
    table1:
      columns:
        date_col:
          sample_values:
          - 2012-05-01
          type: generate_values
        datetime_col:
          sample_values:
          - 2020-11-30 18:29:19.990601
          type: generate_values
        double_col:
          sample_values:
          - 0.89
          type: generate_values
        float_col:
          sample_values:
          - 10.010000228881836
          type: generate_values
        int_col:
          sample_values:
          - 1
          type: generate_values
        long_col:
          sample_values:
          - 2
          type: generate_values
        string_col:
          sample_values:
          - awesome string
          type: generate_values
      num_rows: 10
    table2:
      columns:
        date_col:
          sample_values:
          - 2012-05-01
          type: generate_values
        datetime_col:
          sample_values:
          - 2020-11-30 18:29:19.990601
          type: generate_values
        double_col:
          sample_values:
          - 0.89
          type: generate_values
        float_col:
          sample_values:
          - 10.010000228881836
          type: generate_values
        int_col:
          sample_values:
          - 1
          type: generate_values
        long_col:
          sample_values:
          - 2
          type: generate_values
        string_col:
          sample_values:
          - awesome string
          type: generate_values
      num_rows: 10
    table3:
      columns:
        date_col:
          sample_values:
          - 2012-05-01
          type: generate_values
        datetime_col:
          sample_values:
          - 2020-11-30 18:29:19.990601
          type: generate_values
        double_col:
          sample_values:
          - 0.89
          type: generate_values
        float_col:
          sample_values:
          - 10.010000228881836
          type: generate_values
        int_col:
          sample_values:
          - 1
          type: generate_values
        long_col:
          sample_values:
          - 2
          type: generate_values
        string_col:
          sample_values:
          - awesome string
          type: generate_values
      num_rows: 10
    """
    return expected_config_str_tables


def test_reverse_engineer_tables(sample_df, expected_config_str_tables):
    valid_types_only = sample_df.drop("array_int")
    various_tables = {
        "table1": valid_types_only,
        "table2": valid_types_only,
        "table3": valid_types_only,
    }

    data_fabrication_config = reverse_engineer_tables(various_tables)

    generated_config = yaml.safe_dump(data_fabrication_config)

    assert yaml.safe_load(generated_config) == yaml.safe_load(
        expected_config_str_tables
    )

    mock_generator = MockDataGenerator(instructions=data_fabrication_config)
    mock_generator.generate_all()

    assert {"table1", "table2", "table3"}.issubset(
        set(mock_generator.all_dataframes.keys())
    )

    assert mock_generator.all_dataframes["table1"].shape == (10, 7)
    assert mock_generator.all_dataframes["table2"].shape == (10, 7)
    assert mock_generator.all_dataframes["table3"].shape == (10, 7)
    assert (
        list(mock_generator.all_dataframes["table1"].columns)
        == valid_types_only.columns
    )
    assert (
        list(mock_generator.all_dataframes["table2"].columns)
        == valid_types_only.columns
    )
    assert (
        list(mock_generator.all_dataframes["table3"].columns)
        == valid_types_only.columns
    )
