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
# pylint: disable=redefined-outer-name,logging-fstring-interpolation

"""Contains null check decorator."""

import functools
import logging
from operator import or_
from typing import List, Union

import pandas as pd
import pyspark
import pyspark.sql.functions as f

from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__file__)


COLUMNS_NO_NULLS = "columns_no_nulls"


def _output_has_no_nulls(func, *args, **kwargs):
    """Logic for checking output no nulls."""
    columns_no_nulls, args, kwargs = _get_param_from_arg_and_kwarg(
        COLUMNS_NO_NULLS, *args, **kwargs
    )
    if columns_no_nulls:
        if not isinstance(columns_no_nulls, list):
            columns_no_nulls = [columns_no_nulls]

        def wrapper(*args, **kwargs):
            result_df = func(*args, **kwargs)
            for schema in columns_no_nulls:
                output = schema.pop("output", 0)
                list_of_columns = schema.pop("list_of_columns")
                logger.info(f"The output being checked is: {output}")
                if isinstance(result_df, (list, tuple, dict)):
                    _check_no_null(
                        result_df[output],
                        df_name=output,
                        list_of_columns=list_of_columns,
                    )
                else:
                    _check_no_null(
                        result_df, df_name=output, list_of_columns=list_of_columns
                    )
            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def columns_no_nulls():
    """Checks for nulls in specified columns."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(
            *args, **kwargs,
        ):
            new_func, args, kwargs = _output_has_no_nulls(func, *args, **kwargs,)
            result_df = new_func(*args, **kwargs)
            return result_df

        return wrapper

    return decorate


def _check_no_null(
    df: Union[pd.DataFrame, pyspark.sql.DataFrame],
    df_name: str,
    list_of_columns: List[str],
):
    if isinstance(df, pyspark.sql.DataFrame):
        filter_condition = [f.col(c).isNull() for c in list_of_columns]
        filter_formula = functools.reduce(or_, filter_condition)

        null_count = df.filter(filter_formula).count()
        if null_count > 0:
            logger.info("Found nulls.")
            df.filter(filter_formula).show(truncate=False)
            raise ValueError(f"Found null in one of {list_of_columns} in df {df_name}.")
    elif isinstance(df, pd.DataFrame):
        raise NotImplementedError
    else:
        raise ValueError(f"{df} should be a spark or pandas dataframe.")
