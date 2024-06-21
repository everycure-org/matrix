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

import pytest

from refit.v1.core.input_kwarg_filter import (
    _add_input_kwarg_filter,
    add_input_kwarg_filter,
)


@add_input_kwarg_filter()
def my_node_func_input(df):
    return df


def test_add_input_kwarg_filter_base_spark(sample_df_spark_all_dtypes):
    arg, kwargs = _add_input_kwarg_filter(
        df=sample_df_spark_all_dtypes, kwarg_filter={"df": "int_col != 1"}
    )

    assert kwargs["df"].count() == 0


def test_add_input_kwarg_filter_base_pandas(sample_df_pd_all_dtypes):
    arg, kwargs = _add_input_kwarg_filter(
        df=sample_df_pd_all_dtypes, kwarg_filter={"df": "int_col != 1"}
    )

    assert kwargs["df"].shape[0] == 0


def test_add_input_kwarg_filter_base_pandas_engine_python(sample_df_pd_all_dtypes):
    arg, kwargs = _add_input_kwarg_filter(
        df=sample_df_pd_all_dtypes,
        kwarg_filter={"df": "string_col.str.contains('fo')", "engine": "python"},
    )

    assert kwargs["df"].shape[0] == 1


def test_add_input_kwarg_filter_spark(sample_df_spark_all_dtypes):
    result = my_node_func_input(
        df=sample_df_spark_all_dtypes, kwarg_filter={"df": "int_col != 1"}
    )

    assert result.count() == 0


def test_add_input_kwarg_filter_pandas(sample_df_pd_all_dtypes):
    result = my_node_func_input(
        df=sample_df_pd_all_dtypes, kwarg_filter={"df": "int_col != 1"}
    )

    assert result.shape[0] == 0


def test_add_input_kwarg_filter_no_param(sample_df_spark_all_dtypes):
    """Decorator should not affect function."""
    result = my_node_func_input(df=sample_df_spark_all_dtypes)
    assert result.count() == 1


def test_multi_input(sample_df_spark_all_dtypes):
    @add_input_kwarg_filter()
    def my_node_func(df1, df2):
        return df1, df2, df1.union(df2)

    result1, result2, result3 = my_node_func(
        df1=sample_df_spark_all_dtypes,
        df2=sample_df_spark_all_dtypes,
        kwarg_filter={"df1": "int_col != 1", "df2": "int_col == 1"},
    )

    assert result1.count() == 0
    assert result2.count() == 1
    assert result3.count() == 1


def test_add_input_bad_kwarg(sample_df_spark_all_dtypes):
    with pytest.raises(KeyError, match="Cannot find df1 in kwargs."):
        my_node_func_input(
            df=sample_df_spark_all_dtypes, kwarg_filter={"df1": "int_col != 1"}
        )
