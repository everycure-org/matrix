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
# pylint: disable=logging-fstring-interpolation

"""Decorator to fill nulls."""
import functools
import logging

import pandas as pd
import pyspark

from refit.v1.core.make_list_regexable import _extract_elements_in_list
from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__file__)
FILL_NULLS_KW = "fill_nulls"


def _fill_nulls(func, *args, **kwargs):
    """Fills all nulls with the given value, defaults to 0.

    If column_list specified, fill nulls for the list of columns.
    `column_list` can also take in a list of regex, when
    enable_regex is set to `True`.
    """
    fill_nulls_keys, args, kwargs = _get_param_from_arg_and_kwarg(
        FILL_NULLS_KW, *args, **kwargs
    )

    if fill_nulls_keys:
        enable_regex = fill_nulls_keys.get("enable_regex", False)
        raise_exc = fill_nulls_keys.get("raise_exc", True)

        def wrapper(*args, **kwargs):
            if "df" in kwargs:
                input_df = kwargs["df"]
                _log_dataframe_shape(input_df, "Input")

            result_df = func(*args, **kwargs)
            logger.info("Applying fill nulls to returned result.")
            value = fill_nulls_keys.get("value", 0)

            if enable_regex:
                column_list = _extract_elements_in_list(
                    full_list_of_columns=result_df.columns,
                    list_of_regexes=fill_nulls_keys.get("column_list"),
                    raise_exc=raise_exc,
                )
            else:
                column_list = fill_nulls_keys.get("column_list")
            if isinstance(result_df, pd.DataFrame):
                result_df[column_list] = result_df[column_list].fillna(value)
            elif isinstance(result_df, pyspark.sql.DataFrame):
                result_df = result_df.fillna(value, subset=column_list)
            else:
                raise TypeError(
                    "Decorator only works if function returns a single dataframe."
                )
            _log_dataframe_shape(result_df, "Output")
            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def _log_dataframe_shape(df, label):
    """Log dataframe shape for input or output df."""
    if isinstance(df, pd.DataFrame):
        logger.debug(
            f"{label} DataFrame shape: {df.shape}",
            extra={"fields": {f"{label} DataFrame shape": df.shape}},
        )

    elif isinstance(df, pyspark.sql.DataFrame):
        if logger.isEnabledFor(logging.DEBUG):
            df_count = df.count()
            df_len = len(df.columns)
            logger.debug(
                f"{label} DataFrame shape: ({df_count}, {df_len})",
                extra={"fields": {f"{label} DataFrame shape": (df_count, df_len)}},
            )


def fill_nulls():
    """Fills null values for output dataframe."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(
            *args, **kwargs,
        ):
            new_func, args, kwargs = _fill_nulls(func, *args, **kwargs)
            result_df = new_func(*args, **kwargs)
            return result_df

        return wrapper

    return decorate
