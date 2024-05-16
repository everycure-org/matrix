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

"""Decorator to filter the output of a function."""
import functools
import logging

import pandas as pd
import pyspark

from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__file__)
OUTPUT_FILTER_KW = "output_filter"


def _add_output_filter(  # pylint: disable=missing-return-doc,missing-return-type-doc  # noqa: E501
    func, *args, **kwargs
):
    """Filter args or kwargs.

    Raises:
        KeyError: If kwarg_filter key does not match any kwarg key.
    """
    output_filter, args, kwargs = _get_param_from_arg_and_kwarg(
        OUTPUT_FILTER_KW, *args, **kwargs
    )

    if output_filter:

        def wrapper(*args, **kwargs):
            result_df = func(*args, **kwargs)
            logger.info("Applying filter %s to returned result.", output_filter)
            if isinstance(result_df, pd.DataFrame):
                result_df = result_df.query(output_filter, engine="numexpr")
            elif isinstance(result_df, pyspark.sql.DataFrame):
                result_df = result_df.filter(output_filter)
            else:
                raise TypeError(
                    "Decorator only works if function returns a single dataframe."
                )
            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def add_output_filter():  # pylint: disable=missing-return-type-doc
    """Modifies function definition to include an additional filter kwarg at the output.

    Allows an additional parameter to filter the output without modifying the
    source code itself. Meant to be used with node functions.

    Example usage:
    ::
        @add_output_filter()
        def my_node_func(df):
            return df

        my_node_func(df=df, output_filter="int_col != 1")

    Returns:
        Wrapper function.

    Raises:
        TypeError: If function does not return either a pandas or spark dataframe.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_func, args, kwargs = _add_output_filter(func, *args, **kwargs)

            result_df = new_func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate
