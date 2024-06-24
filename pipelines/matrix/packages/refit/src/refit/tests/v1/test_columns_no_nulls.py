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
# pylint: disable=unexpected-keyword-arg
"""Test cases for `columns_no_nulls` decorator."""
import pytest

from refit.v1.core.columns_no_nulls import columns_no_nulls


def test_base_case(spark_df):
    """Test base case."""

    @columns_no_nulls()
    def func1(x):
        return x

    result = func1(x=spark_df, columns_no_nulls={"list_of_columns": ["int_col"]})
    assert result.count() == 3


def test_multiple_outputs_list(spark_df):
    """Test multiple outputs."""

    @columns_no_nulls()
    def func1(x):
        return [x, x]

    result = func1(
        x=spark_df,
        columns_no_nulls=[
            {"output": 0, "list_of_columns": ["int_col"]},
            {"output": 1, "list_of_columns": ["float_col"]},
        ],
    )
    assert result[0].count() == 3
    assert result[1].count() == 3


def test_multiple_outputs_dict(spark_df):
    """Test multiple outputs."""

    @columns_no_nulls()
    def func1(x):
        return {"x": x, "y": x}

    result = func1(
        x=spark_df,
        columns_no_nulls=[
            {"output": "x", "list_of_columns": ["int_col"]},
            {"output": "y", "list_of_columns": ["float_col"]},
        ],
    )
    assert result["x"].count() == 3
    assert result["y"].count() == 3


def test_multiple_columns(spark_df):
    """Test multiple columns."""

    @columns_no_nulls()
    def func1(x):
        return x

    result = func1(
        x=spark_df, columns_no_nulls={"list_of_columns": ["int_col", "float_col"]}
    )
    assert result.count() == 3


def test_fail_case(spark_df):
    """Test fail case."""

    @columns_no_nulls()
    def func1(x):
        return x

    with pytest.raises(ValueError, match=r"Found null in one of"):
        func1(
            x=spark_df,
            columns_no_nulls={
                "list_of_columns": ["int_col", "float_col", "string_col"]
            },
        )


def test_pandas(pandas_df):
    """Test pandas df."""

    @columns_no_nulls()
    def func1(x):
        return x

    with pytest.raises(NotImplementedError):
        func1(
            x=pandas_df,
            columns_no_nulls={
                "list_of_columns": ["int_col", "float_col", "string_col"]
            },
        )
