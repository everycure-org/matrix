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
import re

import pytest

from refit.v1.core.output_primary_key import add_output_primary_key


@add_output_primary_key()
def my_node_func_input(df):
    return df


# ----- primary key check for spark output dataframe without null check-----
def test_add_output_primary_key_spark_without_null_without_dupe(spark_df,):
    output_primary_key = {"columns": ["int_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_without_null_with_dupe(spark_df,):
    output_primary_key = {"columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_spark_without_null_without_dupe_list1(spark_df,):
    output_primary_key = {"columns": ["int_col", "float_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_without_null_without_dupe_list2(spark_df,):
    output_primary_key = {"columns": ["int_col", "string_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['int_col', 'string_col'\] has null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


# ----- primary key check for spark output dataframe with null check-----
def test_add_output_primary_key_spark_with_null_without_dupe(spark_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_with_null_without_dupe_with_list(spark_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col", "string_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_with_null_with_dupe(spark_df, caplog):
    output_primary_key = {"nullable": True, "columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert (
        "Primary key constraint violation for index - 0: 3 row(s) have "
        + "duplicate value(s) in columns ['float_col']. First offending rows:"
        in caplog.text
    )
    assert "int_col  float_col      string_col  cnt" in caplog.text
    assert "0        1        2.0  awesome string    3" in caplog.text
    assert "1        2        2.0            None    3" in caplog.text
    assert "2        3        2.0     hello world    3" in caplog.text


def test_add_output_primary_key_spark_with_null_val(spark_df, caplog):
    output_primary_key = {"columns": ["string_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['string_col'\] has null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert (
        "Primary key constraint violation for index - 0: 1 row(s) have null "
        + "value(s) in columns ['string_col']. First offending rows:"
        in caplog.text
    )
    assert "int_col  float_col string_col" in caplog.text
    assert "0        2        2.0       None" in caplog.text


# ----- primary key check for Pandas output dataframe without null check-----
def test_add_output_primary_key_pandas_without_null(pandas_df):
    output_primary_key = {"columns": ["int_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_without_null_with_dupe(pandas_df,):
    output_primary_key = {"columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_pandas_without_null_with_list(pandas_df,):
    output_primary_key = {"columns": ["int_col", "float_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


# ----- primary key check for Pandas output dataframe with null check-----
def test_add_output_primary_key_pandas_with_null(pandas_df):
    output_primary_key = {"nullable": True, "columns": ["int_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_with_null_with_list(pandas_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col", "float_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_with_null_with_dupe(pandas_df, caplog):
    output_primary_key = {"nullable": True, "columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert (
        "Primary key constraint violation for index - 0: 3 row(s) have "
        + "duplicate value(s) in columns ['float_col']. First offending rows:"
        in caplog.text
    )
    assert "float_col  int_col string_col" in caplog.text
    assert "0        1.0        1        foo" in caplog.text
    assert "1        1.0        2     blabla" in caplog.text
    assert "2        1.0        3       None" in caplog.text


def test_add_output_primary_key_pandas_with_null_val(pandas_df):
    output_primary_key = {"nullable": True, "columns": ["string_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_with_null_val_not_nullable(pandas_df, caplog):
    output_primary_key = {"nullable": False, "columns": ["string_col"]}

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Primary key columns for index - 0, ['string_col'] has null values."
        ),
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert (
        "Primary key constraint violation for index - 0: 1 row(s) have null value(s) "
        + "in columns ['string_col']. First offending rows:"
        in caplog.text
    )
    assert "float_col  int_col string_col" in caplog.text
    assert "2        1.0        3       None" in caplog.text
