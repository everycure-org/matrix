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
"""Decorator to check for primary key constraints on dataframes."""
import functools
import logging

from refit.v1.core.output_primary_key import _duplicate_and_null_check
from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__file__)


OUTPUT_PRIMARY_KEY = "output_primary_key"
INPUT_PRIMARY_KEY = "input_primary_key"


def _input_primary_key(input_pk_confs, *args, **kwargs):
    """Check input tables primary key constraints."""
    if input_pk_confs:
        if not args and not kwargs:
            raise ValueError(
                f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed."
            )

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                f"Please use either args or kwargs with `{INPUT_PRIMARY_KEY}`."
            )

        all_args = args if len(args) > 0 else kwargs

        if len(all_args) > 1:
            for pk_conf in input_pk_confs:
                input_dataframe_index = pk_conf.get("index")

                _duplicate_and_null_check(
                    result_df=all_args[input_dataframe_index],
                    df_name=input_dataframe_index,
                    columns_list=pk_conf.get("columns"),
                    nullable=pk_conf.get("nullable"),
                )
        else:
            # Handle the following case:
            # @primary_key()
            # def node(x):
            #     ...
            #
            # # arg case -> default index to 0
            # node(x, input_primary_key={"columns": ["ABC"]})
            # # kwarg case -> default index to x
            # node(x=x, input_primary_key={"columns": ["ABC"]})
            default_key = 0
            if isinstance(all_args, dict):
                default_key = list(all_args.keys())[0]

            pk_conf = input_pk_confs[0]
            _duplicate_and_null_check(
                result_df=all_args[default_key],
                columns_list=pk_conf.get("columns"),
                nullable=pk_conf.get("nullable"),
            )


def _output_primary_key(output_pk_confs, result):
    """Check output tables primary key constraints."""
    if output_pk_confs:
        if isinstance(result, (dict, list, tuple)):
            for pk_conf in output_pk_confs:
                _duplicate_and_null_check(
                    result_df=result[pk_conf.get("index")],
                    columns_list=pk_conf.get("columns"),
                    nullable=pk_conf.get("nullable"),
                )

        else:
            if len(output_pk_confs) != 1:
                raise ValueError(
                    f"`{OUTPUT_PRIMARY_KEY}` should have just one configuration."
                )

            pk_conf = output_pk_confs[0]
            _duplicate_and_null_check(
                result_df=result,
                columns_list=pk_conf.get("columns"),
                nullable=pk_conf.get("nullable"),
            )


def primary_key():
    """Check the primary key constraints on dataframes."""

    def _decorate(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            output_pk_confs, args, kwargs = _get_param_from_arg_and_kwarg(
                OUTPUT_PRIMARY_KEY, *args, **kwargs
            )
            input_pk_confs, args, kwargs = _get_param_from_arg_and_kwarg(
                INPUT_PRIMARY_KEY, *args, **kwargs
            )

            if isinstance(input_pk_confs, dict):
                input_pk_confs = [input_pk_confs]
            if isinstance(output_pk_confs, dict):
                output_pk_confs = [output_pk_confs]

            # Handling inputs
            _input_primary_key(input_pk_confs, *args, **kwargs)

            result = func(*args, **kwargs)

            # Handling outputs
            _output_primary_key(output_pk_confs, result)

            return result

        return _wrapper

    return _decorate
