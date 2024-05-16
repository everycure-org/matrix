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
"""Tests fill nulls filter."""
import logging

import pandas as pd

from refit.v1.core.fill_nulls import _fill_nulls, fill_nulls


def dummy_func_df(df):
    return df


@fill_nulls()
def my_node_func_input(df):
    return df


def test_fill_nulls_for_boolean_spark(spark_nulls_df, caplog):
    fill_nulls = {"value": True, "enable_regex": True, "column_list": [".*"]}

    new_func, args, kwargs = _fill_nulls(
        dummy_func_df, **{"df": spark_nulls_df, "fill_nulls": fill_nulls}
    )

    with caplog.at_level(logging.DEBUG):
        result_df = new_func(*args, **kwargs)
    assert "Input DataFrame shape: (4, 4)" in caplog.text
    assert "Output DataFrame shape: (4, 4)" in caplog.text
    result = [x[0] for x in result_df.select("bool_col").collect()]
    float_col = [x[0] for x in result_df.select("float_col").collect()]
    assert result == [True, False, True, False]
    # spark only fills the columns which match with the type of value sent as input.
    assert float_col == [None, 2.0, 2.0, None]
    assert result_df.dtypes == [
        ("int_col", "int"),
        ("float_col", "float"),
        ("string_col", "string"),
        ("bool_col", "boolean"),
    ]


def test_fill_nulls_for_number_spark(spark_nulls_df):
    fill_nulls = {"value": 0, "enable_regex": True, "column_list": [".*"]}

    new_func, args, kwargs = _fill_nulls(
        dummy_func_df, **{"df": spark_nulls_df, "fill_nulls": fill_nulls}
    )
    result_df = new_func(*args, **kwargs)

    float_col = [x[0] for x in result_df.select("float_col").collect()]
    int_col = [x[0] for x in result_df.select("int_col").collect()]
    assert float_col == [0.0, 2.0, 2.0, 0.0]
    assert int_col == [1, 2, 3, 0]
    assert result_df.dtypes == [
        ("int_col", "int"),
        ("float_col", "float"),
        ("string_col", "string"),
        ("bool_col", "boolean"),
    ]


def test_fill_nulls_for_subset_columns_spark(spark_nulls_df):
    fill_nulls = {"value": 0, "column_list": ["int_col"]}

    new_func, args, kwargs = _fill_nulls(
        dummy_func_df, **{"df": spark_nulls_df, "fill_nulls": fill_nulls}
    )
    result_df = new_func(*args, **kwargs)

    int_col = [x[0] for x in result_df.select("int_col").collect()]
    assert int_col == [1, 2, 3, 0]


def test_fill_nulls_for_boolean_column_pandas(pandas_nulls_df, caplog):
    fill_nulls = {"value": False, "column_list": ["bool_col"]}

    new_func, args, kwargs = _fill_nulls(
        dummy_func_df, **{"df": pandas_nulls_df, "fill_nulls": fill_nulls}
    )
    with caplog.at_level(logging.DEBUG):
        result_df = new_func(*args, **kwargs)
    assert "Input DataFrame shape: (4, 4)" in caplog.text
    assert "Output DataFrame shape: (4, 4)" in caplog.text
    assert result_df["bool_col"].tolist() == [True, False, False, True]


def test_fill_nulls_all_columns_pandas(pandas_nulls_df):
    fill_nulls = {"value": 0, "enable_regex": True, "column_list": [".*"]}

    result = my_node_func_input(df=pandas_nulls_df, fill_nulls=fill_nulls)

    expected = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 0],
            "float_col": [0.0, 2.0, 2.0, 0.0],
            "string_col": ["foo", 0, "sample", "awesome"],
            "bool_col": [True, 0, 0, True],
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
