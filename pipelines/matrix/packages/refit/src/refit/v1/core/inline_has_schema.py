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
# pylint: disable=missing-return-type-doc
"""Contains the inline style `has_schema`.

Just the way you remember it.
"""

from inspect import getfullargspec
from typing import Mapping

from boltons.funcutils import wraps

from refit.v1.core.has_schema import _check_schema


def has_schema(
    schema: Mapping[str, str],
    output: int = None,
    df: str = None,
    allow_subset: bool = True,
    raise_exc: bool = True,
    relax: bool = True,
):
    """Check if the input or output dataframe has the expected schema.

    Defaults to checking schema of the first output dataframe.

    Args:
        schema: The schema of the specified input or output.
        output: The index of the output. Defaults to 0.
        df: The string name of the input dataframe.
        allow_subset: Whether to raise exception if provided schema is a subset of
            the actual dataframe.
        raise_exc: Whether to raise an exception or just log the warning.
        relax: Allows for loose equivalency of dtypes for spark dataframes. i.e.
            float and double, int and long, date and timestamp.

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(
            *args,
            output=output,
            df_name=df,
            allow_subset=allow_subset,
            raise_exc=raise_exc,
            relax=relax,
            **kwargs,
        ):
            # default 0 messes assert, so using this pattern
            if df_name and output is not None:
                raise ValueError("Please supply only df or output.")

            if df_name:

                argspec = getfullargspec(func)
                df_index = argspec.args.index(df_name)
                df = args[df_index]

                _check_schema(
                    df=df,
                    df_name=df_name,
                    expected_schema=schema,
                    allow_subset=allow_subset,
                    raise_exc=raise_exc,
                    relax=relax,
                )

                results = func(*args, **kwargs)

            else:

                output = output or 0

                results = func(*args, **kwargs)

                # split to not modify the result
                if isinstance(results, list):
                    _check_schema(
                        df=results[output],
                        df_name=output,
                        expected_schema=schema,
                        allow_subset=allow_subset,
                        raise_exc=raise_exc,
                        relax=relax,
                    )
                else:
                    _check_schema(
                        df=results,
                        df_name=output,
                        expected_schema=schema,
                        allow_subset=allow_subset,
                        raise_exc=raise_exc,
                        relax=relax,
                    )

            return results

        return _wrapper

    return _decorate
