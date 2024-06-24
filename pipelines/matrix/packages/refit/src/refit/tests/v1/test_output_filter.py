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
"""Tests output filter."""
import pytest

from refit.v1.core.output_filter import _add_output_filter, add_output_filter


def dummy_func_df(df):
    return df


def dummy_func_df_to_str(df):
    return "dummy_string"


@add_output_filter()
def my_node_func_input(df):
    return df


def test_add_output_filter_base_spark(sample_df_spark_all_dtypes):
    new_func, args, kwargs = _add_output_filter(
        dummy_func_df,
        **{"df": sample_df_spark_all_dtypes, "output_filter": "int_col != 1"}
    )

    result = new_func(*args, **kwargs)

    assert result.count() == 0


def test_add_output_filter_base_pandas(sample_df_pd_all_dtypes):
    new_func, args, kwargs = _add_output_filter(
        dummy_func_df, df=sample_df_pd_all_dtypes, output_filter="int_col != 1"
    )

    result = new_func(*args, **kwargs)

    assert result.shape[0] == 0


def test_add_output_filter_pandas(sample_df_pd_all_dtypes):
    result = my_node_func_input(sample_df_pd_all_dtypes, output_filter="int_col != 1")

    assert result.shape[0] == 0


def test_add_output_filter_spark(sample_df_spark_all_dtypes):
    result = my_node_func_input(
        df=sample_df_spark_all_dtypes, output_filter="int_col != 1"
    )

    assert result.count() == 0


def test_output_operations_no_param(sample_df_spark_all_dtypes):
    """Decorator should not affect function."""
    result = my_node_func_input(df=sample_df_spark_all_dtypes)
    assert result.count() == 1


def test_add_output_filter_invalid_df(sample_df_pd_all_dtypes):
    new_func, args, kwargs = _add_output_filter(
        dummy_func_df_to_str, df=sample_df_pd_all_dtypes, output_filter="1=1",
    )
    with pytest.raises(
        TypeError, match="Decorator only works if function returns a single dataframe."
    ):
        new_func(*args, **kwargs)
