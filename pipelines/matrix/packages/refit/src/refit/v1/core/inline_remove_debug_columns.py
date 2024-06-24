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
# pylint: disable=redefined-outer-name,line-too-long
# flake8: noqa
"""Contains the inline style `remove_debug_columns`."""

from boltons.funcutils import wraps

from refit.v1.core.remove_debug_columns import _remove_debug_columns


def remove_input_debug_columns(
    prefix: str = "_", remove_input_debug_columns: bool = False,
):  # pylint: disable=missing-return-type-doc
    """Remove columns with a given prefix from all dataframe inputs configurable inline.

    Args:
        prefix: The column prefix to remove. Defaults to underscore "_".
        remove_input_debug_columns: Whether to remove input debug columns.
            Defaults to False.

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            if not remove_input_debug_columns:
                return func(*args, **kwargs)

            all_args = []
            for arg in args:
                arg = _remove_debug_columns(arg, prefix)
                all_args.append(arg)

            all_kwargs = {}
            for key, value in kwargs.items():
                value = _remove_debug_columns(value, prefix)
                all_kwargs[key] = value

            result = func(*all_args, **all_kwargs)

            return result

        return _wrapper

    return _decorate


def remove_output_debug_columns(
    prefix: str = "_", remove_output_debug_columns: bool = False,
):  # pylint: disable=missing-return-type-doc
    """Remove columns with a given prefix from all dataframe outputs configurable inline.

    Args:
        prefix: The column prefix to remove. Defaults to underscore "_".
        remove_output_debug_columns: Whether to remove output debug columns.
            Defaults to False.

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):

            result = func(*args, **kwargs)

            if remove_output_debug_columns:
                if isinstance(result, list):
                    result = [_remove_debug_columns(df, prefix) for df in result]
                else:
                    result = _remove_debug_columns(result, prefix)

            return result

        return _wrapper

    return _decorate
