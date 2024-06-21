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
# pylint: disable=missing-return-type-doc,redefined-outer-name
"""Contains the Inline style `primary_key`.

Just the way you remember it.
"""

from inspect import getfullargspec
from typing import List

from boltons.funcutils import wraps

from refit.v1.core.output_primary_key import _duplicate_and_null_check


def primary_key(
    primary_key: List[str], nullable: bool = False, output: int = None, df: str = None,
):
    """Checks the primary key validation for list of columns of a dataframe.

    Args:
        primary_key: Columns list to perform primary key check validation either on
                    input or output dataframe
        df: The string name of the input dataframe.
        output: The index of the output. Defaults to 0.
        nullable: Whether to allow nulls or not in primary/composite key.

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(
            *args,
            df_name=df,
            primary_key=primary_key,
            nullable=nullable,
            output=output,
            **kwargs,
        ):
            # default 0 messes assert, so using this pattern
            if df_name and output is not None:
                raise ValueError("Please supply either df or output.")

            if df_name:

                argspec = getfullargspec(func)
                df_index = argspec.args.index(df_name)
                df = args[df_index]

                _duplicate_and_null_check(
                    result_df=df,
                    columns_list=primary_key,
                    nullable=nullable,
                    df_name=df_name,
                )

                results = func(*args, **kwargs)

            else:

                output = output or 0

                results = func(*args, **kwargs)

                # split to not modify the result
                if isinstance(results, list):
                    _duplicate_and_null_check(
                        result_df=results[output],
                        columns_list=primary_key,
                        nullable=nullable,
                        df_name=output,
                    )
                else:
                    _duplicate_and_null_check(
                        result_df=results,
                        columns_list=primary_key,
                        nullable=nullable,
                        df_name=output,
                    )

            return results

        return _wrapper

    return _decorate
