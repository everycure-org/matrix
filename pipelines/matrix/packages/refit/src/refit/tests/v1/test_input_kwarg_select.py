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

from refit.v1.core.input_kwarg_select import (
    _add_input_kwarg_select,
    add_input_kwarg_select,
)


@add_input_kwarg_select()
def my_node_func_input(df):
    return df


def test_add_input_kwarg_select_base_spark(sample_df_spark_all_dtypes):
    arg, kwargs = _add_input_kwarg_select(
        df=sample_df_spark_all_dtypes,
        kwarg_select={"df": ["length(string_col) as string_col_length"]},
    )

    assert kwargs["df"].columns == ["string_col_length"]


def test_add_input_kwarg_select_spark(sample_df_spark_all_dtypes):
    result = my_node_func_input(
        df=sample_df_spark_all_dtypes,
        kwarg_select={"df": ["length(string_col) as string_col_length"]},
    )

    assert result.columns == ["string_col_length"]


def test_add_input_kwarg_select_no_param(sample_df_spark_all_dtypes):
    """Decorator should not affect function."""
    result = my_node_func_input(df=sample_df_spark_all_dtypes)
    assert result.columns == sample_df_spark_all_dtypes.columns


def test_multi_input(sample_df_spark_all_dtypes):
    @add_input_kwarg_select()
    def my_node_func(df1, df2):
        return df1, df2, df1.union(df2)

    result1, result2, result3 = my_node_func(
        df1=sample_df_spark_all_dtypes,
        df2=sample_df_spark_all_dtypes,
        kwarg_select={
            "df1": ["length(string_col) as string_col_length"],
            "df2": ["length(string_col) as string_col_length"],
        },
    )

    assert result1.columns == ["string_col_length"]
    assert result2.columns == ["string_col_length"]
    assert result3.columns == ["string_col_length"]


def test_add_input_bad_kwarg(sample_df_spark_all_dtypes):
    with pytest.raises(KeyError, match="Cannot find df1 in kwargs."):
        my_node_func_input(
            df=sample_df_spark_all_dtypes,
            kwarg_select={"df1": ["length(string_col) as string_col_length"]},
        )


def test_add_input_wrong_kwarg_type(sample_df_pd_all_dtypes):
    with pytest.raises(
        KeyError, match="Cannot apply select to df as it is a `pandas.DataFrame`"
    ):
        my_node_func_input(
            df=sample_df_pd_all_dtypes,
            kwarg_select={"df": ["length(string_col) as string_col_length"]},
        )
