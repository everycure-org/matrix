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

"""Decorator to select dataframe at input of function."""
import functools
import logging

import pandas as pd
import pyspark
from pyspark.sql import functions as f

from refit.v1.internals import _get_param_from_arg_and_kwarg

INPUT_SELECT_KW = "kwarg_select"


def _add_input_kwarg_select(  # pylint: disable=missing-return-doc,missing-return-type-doc  # noqa: E501
    *args, **kwargs
):
    """Select args or kwargs.

    Raises:
        KeyError: If select key does not match any kwarg key.
    """
    input_select, args, kwargs = _get_param_from_arg_and_kwarg(
        INPUT_SELECT_KW, *args, **kwargs
    )

    if input_select:
        for key, select_expressions in input_select.items():
            logging.info(
                "Applying selection %s to keyword arg %s.", select_expressions, key
            )
            available_kwargs = list(
                filter(lambda x: isinstance(x, pyspark.sql.DataFrame), kwargs.keys())
            )

            if key not in kwargs:
                raise KeyError(
                    f"Cannot find {key} in kwargs. "
                    f"Available kwargs: {available_kwargs}"
                )

            if isinstance(kwargs[key], pd.DataFrame):
                raise KeyError(
                    f"Cannot apply select to {key} as it is a `pandas.DataFrame`."
                    f"Available kwargs: {available_kwargs}"
                )
            if isinstance(kwargs[key], pyspark.sql.DataFrame):
                kwargs[key] = kwargs[key].select(
                    *[f.expr(select_expr) for select_expr in select_expressions]
                )

    return args, kwargs


def add_input_kwarg_select():  # pylint: disable=missing-return-type-doc
    """Modifies function definition to include an additional select kwarg at the input.

    Allows the addition of an additional parameter to select the inputs based on
    kwargs without modifying the source code itself. Meant to be used with node
    functions.

    Example usage:
    ::
        @add_input_kwarg_select()
        def my_node_func(df):
            return df

        my_node_func(df=df, kwarg_select={"df": ["len(str_col)"]})

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            args, kwargs = _add_input_kwarg_select(*args, **kwargs)

            result_df = func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate
