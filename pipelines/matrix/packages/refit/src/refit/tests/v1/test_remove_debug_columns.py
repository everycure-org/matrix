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
"""Tests."""

import pandas as pd
import pytest

from refit.v1.core.remove_debug_columns import (
    remove_input_debug_columns,
    remove_output_debug_columns,
)

PREFIX = "_"


@pytest.fixture
def sample_el():
    """Sample elasticsearch response."""
    return [
        {
            "_index": "nhs_demo",
            "_type": "_doc",
            "_id": "dLqZPnAB79C17UGFXPlq",
            "_score": 23.907246,
            "_source": {
                "site_id": "4",
                "site_name": "#2180632 natl ry sedol #ca7 canadyan com isin co",
            },
        },
        {
            "_index": "nhs_demo",
            "_type": "_doc",
            "_id": "l7qZPnAB79C17UGFXPlr",
            "_score": 23.29108,
            "_source": {
                "site_id": "39",
                "site_name": "#ca7 com sedol co canadhan ry #2180632 natl isin",
            },
        },
    ]


@pytest.fixture
def sample_el_df_pd(sample_el):
    df = pd.DataFrame(sample_el)
    df["non_hidden"] = 1
    return df


@pytest.fixture
def sample_el_df_spark(sample_el_df_pd, spark):
    return spark.createDataFrame(sample_el_df_pd)


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func1(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 0
    return df1


@remove_input_debug_columns()
def some_func2(df1, df2, dict1):
    debug_cols1 = [x for x in df1.columns if x.startswith(PREFIX)]
    debug_cols2 = [x for x in df2.columns if x.startswith(PREFIX)]
    assert len(debug_cols1) == 0
    assert len(debug_cols2) == 0
    assert dict1["key"] == "value"
    assert "non_hidden" in df1.columns
    assert "non_hidden" in df2.columns


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func3(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    df1["new_col1"] = 1
    df1["_new_col2"] = 2
    assert len(debug_cols) == 0
    return df1


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func4(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    df1["new_col1"] = 1
    df1["_new_col2"] = 2
    assert len(debug_cols) == 0
    return df1


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func5(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 5
    return df1


def test_input_remove_input_true_output_false_pd(sample_el_df_pd):
    """Test remove_input_debug_columns with remove_output_debug_columns
    set to ignore."""
    some_func1(sample_el_df_pd, remove_input_debug_columns=True)


def test_input_remove_input_true_multiple_pd(sample_el_df_pd):
    """Test remove_input_debug_columns with multiple input pandas dfs."""
    some_func2(
        sample_el_df_pd,
        sample_el_df_pd,
        {"key": "value"},
        remove_input_debug_columns=True,
    )


def test_remove_debug_cols_pd_input_true_output_false(sample_el_df_pd):
    """Test remove_input_debug_columns and remove_output_debug_columns
    with remove_output_debug_columns set to ignore."""
    df1 = some_func3(df1=sample_el_df_pd, remove_input_debug_columns=True)
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 1


def test_remove_debug_cols_pd_input_true_output_true(sample_el_df_pd):
    """Test remove_input_debug_columns and remove_output_debug_columns."""
    df1 = some_func4(
        df1=sample_el_df_pd,
        remove_output_debug_columns=True,
        remove_input_debug_columns=True,
    )
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 0


def test_input_remove_spark_input_true_output_false(sample_el_df_spark):
    """Test remove_input_debug_columns on spark datasets with
    remove_output_debug_columns set to ignore"""
    some_func1(sample_el_df_spark, remove_input_debug_columns=True)


def test_input_remove_input_true_multiple_spark(sample_el_df_spark):
    """Test remove_input_debug_columns with multiple input spark dfs."""
    some_func2(
        df1=sample_el_df_spark,
        df2=sample_el_df_spark,
        dict1={"key": "value"},
        remove_input_debug_columns=True,
    )


def test_remove_debug_cols_spark_input_true_output_false(sample_el_df_spark):
    """Test remove_input_debug_columns and remove_output_debug_columns on
    spark df with remove_output_debug_columns set to ignore."""
    df = some_func1(sample_el_df_spark, remove_input_debug_columns=True)
    cols_start_prefix = [column for column in df.columns if column.startswith(PREFIX)]
    assert cols_start_prefix == []


def test_remove_debug_cols_spark_input_false_output_false(sample_el_df_spark):
    """Test remove_input_debug_columns and remove_output_debug_columns on
    spark df with remove_input_debug_columns and remove_output_debug_columns
    set to ignore."""
    df = some_func5(sample_el_df_spark)
    debug_cols = [x for x in df.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 5
