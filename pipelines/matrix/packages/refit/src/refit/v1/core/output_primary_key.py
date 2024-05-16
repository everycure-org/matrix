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
# pylint: disable=missing-return-doc, missing-return-type-doc, line-too-long

"""Decorator to check primary key of function output."""
import functools
import logging
from functools import reduce
from typing import List, Union

import pandas as pd
import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window

from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__name__)

OUTPUT_PRIMARY_KEY = "output_primary_key"


def _add_output_primary_key(func, *args, **kwargs):
    """Filter args or kwargs.

    Raises:
        KeyError: If kwarg_filter key does not match any kwarg key.
    """
    output_primary_key, args, kwargs = _get_param_from_arg_and_kwarg(
        OUTPUT_PRIMARY_KEY, *args, **kwargs
    )

    if output_primary_key:
        columns_list = output_primary_key["columns"]
        nullable = output_primary_key.get("nullable", False)

        def wrapper(*args, **kwargs):
            result_df = func(*args, **kwargs)

            logger.info(
                "Checking primary key validation on output dataframe - "
                "Non Duplicate Check & Not Null Check",
            )

            _duplicate_and_null_check(result_df, columns_list, nullable)

            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def add_output_primary_key():
    """Primary key check function decorator.

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_func, args, kwargs = _add_output_primary_key(func, *args, **kwargs)
            result_df = new_func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate


def _duplicate_and_null_check(
    result_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    columns_list: List[str],
    nullable: bool,
    df_name: str = "0",
):
    """Duplicate and nullable check on primary key columns of a dataframe."""
    if isinstance(result_df, pyspark.sql.DataFrame):
        if not nullable:
            _check_spark_df_for_nulls(columns_list, result_df, df_name=df_name)
        _check_spark_df_for_duplicates(columns_list, result_df, df_name=df_name)

    elif isinstance(result_df, pd.DataFrame):
        if not nullable:
            _check_pandas_df_for_nulls(columns_list, result_df, df_name=df_name)
        _check_pandas_df_for_duplicates(columns_list, result_df, df_name=df_name)

    else:
        raise ValueError(
            f"Error checking index {df_name}: `result_df` should be of "
            "type pandas or spark dataframe."
        )


def _check_pandas_df_for_duplicates(columns_list, result_df, df_name):
    duplicate_rows = result_df.duplicated(subset=columns_list, keep=False)
    duplicate_count = duplicate_rows.sum()
    if duplicate_count > 0:
        offending_rows = result_df[duplicate_rows].sort_values(by=columns_list).head(10)
        _log_and_raise_primary_key_violation_duplicates(
            columns_list=columns_list,
            duplicate_count=duplicate_count,
            offending_rows=offending_rows,
            df_name=df_name,
        )


def _check_pandas_df_for_nulls(columns_list, result_df, df_name):
    null_rows = result_df[columns_list].isnull().any(axis="columns")
    null_count = null_rows.sum()
    if null_count > 0:
        offending_rows = result_df[null_rows].head(10)
        _log_and_raise_primary_key_violation_nulls(
            columns_list=columns_list,
            null_count=null_count,
            offending_rows=offending_rows,
            df_name=df_name,
        )


def _check_spark_df_for_duplicates(columns_list, result_df, df_name):
    window_from_columns_list = Window.partitionBy(*columns_list)
    result_df = result_df.withColumn("cnt", f.count("*").over(window_from_columns_list))
    duplicate_rows = result_df.filter(f.col("cnt") > 1)
    duplicate_count = duplicate_rows.count()
    if duplicate_count > 0:
        offending_rows = duplicate_rows.sort(*columns_list).limit(10).toPandas()
        _log_and_raise_primary_key_violation_duplicates(
            columns_list=columns_list,
            duplicate_count=duplicate_count,
            offending_rows=offending_rows,
            df_name=df_name,
        )


def _check_spark_df_for_nulls(columns_list, result_df, df_name):
    null_rows = result_df.filter(
        reduce(lambda a, b: a | b, [f.col(column).isNull() for column in columns_list],)
    )
    null_count = null_rows.count()
    if null_count > 0:
        offending_rows = null_rows.limit(10).toPandas()
        _log_and_raise_primary_key_violation_nulls(
            columns_list=columns_list,
            null_count=null_count,
            offending_rows=offending_rows,
            df_name=df_name,
        )


def _log_and_raise_primary_key_violation_duplicates(
    columns_list, duplicate_count, offending_rows, df_name
):
    logger.error(
        "Primary key constraint violation for index - %s: %s row(s) "
        "have duplicate value(s) in columns %s. First offending rows: \n%s",
        df_name,
        duplicate_count,
        columns_list,
        offending_rows,
    )
    raise TypeError(
        f"Primary key columns for index - {df_name}, {columns_list} "
        "has duplicate values."
    )


def _log_and_raise_primary_key_violation_nulls(
    columns_list, null_count, offending_rows, df_name
):
    logger.error(
        "Primary key constraint violation for index - %s: %s row(s) have "
        "null value(s) in columns %s. First offending rows: \n%s",
        df_name,
        null_count,
        columns_list,
        offending_rows,
    )
    raise TypeError(
        f"Primary key columns for index - {df_name}, {columns_list} has null values."
    )
