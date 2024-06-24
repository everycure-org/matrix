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

"""Decorator to filter dataframe at input of function."""
import functools
import logging

import pandas as pd
import pyspark

from refit.v1.internals import _get_param_from_arg_and_kwarg

INPUT_FILTER_KW = "kwarg_filter"


def _add_input_kwarg_filter(  # pylint: disable=missing-return-doc,missing-return-type-doc  # noqa: E501
    *args, **kwargs
):
    """Filter args or kwargs.

    Raises:
        KeyError: If kwarg_filter key does not match any kwarg key.
    """
    input_filter, args, kwargs = _get_param_from_arg_and_kwarg(
        INPUT_FILTER_KW, *args, **kwargs
    )

    if input_filter:
        engine = input_filter.pop("engine", "numexpr")
        for key, filter_expr in input_filter.items():
            logging.info("Applying filter %s to keyword arg %s.", filter_expr, key)

            if key not in kwargs:
                raise KeyError(
                    f"Cannot find {key} in kwargs. "
                    f"Available kwargs: {list(kwargs.keys())}"
                )

            if isinstance(kwargs[key], pd.DataFrame):
                kwargs[key] = kwargs[key].query(filter_expr, engine=engine)
            if isinstance(kwargs[key], pyspark.sql.DataFrame):
                kwargs[key] = kwargs[key].filter(filter_expr)

    return args, kwargs


def add_input_kwarg_filter():  # pylint: disable=missing-return-type-doc
    """Modifies function definition to include an additional filter kwarg at the input.

    Allows the additional of an additional parameter to filter the inputs based on
    kwargs without modifying the source code itself. Meant to be used with node
    functions.

    Example usage:
    ::
        @add_input_kwarg_filter()
        def my_node_func(df):
            return df

        my_node_func(df=df, kwarg_filter={"df": "int_col != 1"})

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            args, kwargs = _add_input_kwarg_filter(*args, **kwargs)

            result_df = func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate
