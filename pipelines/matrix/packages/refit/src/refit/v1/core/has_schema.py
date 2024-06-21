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

"""Decorator to handle input dataframe schema validation."""
import functools
import logging
from typing import Mapping, Union

import pandas as pd
import pyspark

from refit.v1.internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__file__)


INPUT_HAS_SCHEMA_KEY = "input_has_schema"
OUTPUT_HAS_SCHEMA_KEY = "output_has_schema"


def _input_has_schema_node(*args, **kwargs):
    """Checks the schema of ``input`` according to given ``expected_schema`.

    Dictionary format expected
    ::
       {
            "input_has_schema": {
                "expected_schema": {"col_name1": "int", "col_name2": "string"},
                "allow_subset": True,
                "raise_exc": True,
                "relax": True,
            }
        }
    """
    has_schema_keys, args, kwargs = _get_param_from_arg_and_kwarg(
        INPUT_HAS_SCHEMA_KEY, *args, **kwargs
    )
    if has_schema_keys:
        if not isinstance(has_schema_keys, list):
            has_schema_keys = [has_schema_keys]

        for has_schema_key in has_schema_keys:
            dataframe_name = has_schema_key.pop("df")
            logger.info(f"The input being checked is: {dataframe_name}")
            df = kwargs[dataframe_name]

            _check_schema(
                df=df, df_name=dataframe_name, **has_schema_key,
            )

    return args, kwargs


def _output_has_schema_node(func, *args, **kwargs):
    """Checks the schema of ``output`` according to given ``expected_schema`.

    Dictionary format expected
    ::
       {
            "output_has_schema": {
                "expected_schema": {"col_name1": "int", "col_name2": "string"},
                "allow_subset": True,
                "raise_exc": True,
                "relax": True,
            }
        }
    """
    output_schema, args, kwargs = _get_param_from_arg_and_kwarg(
        OUTPUT_HAS_SCHEMA_KEY, *args, **kwargs
    )
    if output_schema:
        if not isinstance(output_schema, list):
            output_schema = [output_schema]

        def wrapper(*args, **kwargs):
            result_df = func(*args, **kwargs)
            for schema in output_schema:
                output = schema.pop("output", 0)
                logger.info(f"The output being checked is: {output}")
                if isinstance(result_df, (list, tuple, dict)):
                    _check_schema(result_df[output], df_name=output, **schema)
                else:
                    _check_schema(result_df, df_name=output, **schema)
            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def has_schema():
    """Checks the schema of ``input`` or ``output`` according to given ``schema``."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(
            *args, **kwargs,
        ):
            args, kwargs = _input_has_schema_node(*args, **kwargs,)
            new_func, args, kwargs = _output_has_schema_node(func, *args, **kwargs,)
            result_df = new_func(*args, **kwargs)
            return result_df

        return wrapper

    return decorate


def _relax_dtype(
    relax_conversion: Mapping[str, str],
    actual_schema: Mapping[str, str],
    expected_schema: Mapping[str, str],
):
    actual_schema = {
        col: relax_conversion.get(dtype, dtype) for col, dtype in actual_schema.items()
    }
    expected_schema = {
        col: relax_conversion.get(dtype, dtype)
        for col, dtype in expected_schema.items()
    }

    return actual_schema, expected_schema


def _check_schema(  # noqa: C901
    df: Union[pd.DataFrame, pyspark.sql.DataFrame],
    expected_schema: Mapping[str, str],
    df_name: str,
    allow_subset: bool = True,
    raise_exc: bool = True,
    relax: bool = True,
):
    if isinstance(df, pd.DataFrame):
        actual_schema = {col: df[col].dtype.name for col in df.columns}
        if relax:
            RELAX_CONVERSION = {  # pylint: disable=invalid-name
                "float16": "numeric",
                "float32": "numeric",
                "float64": "numeric",
                "float": "numeric",
                "int8": "numeric",
                "int16": "numeric",
                "int32": "numeric",
                "int64": "numeric",
                "int": "numeric",
                "uint8": "numeric",
                "uint16": "numeric",
                "uint32": "numeric",
                "uint64": "numeric",
                "datetime64[ns]": "datetime",
                "datetime64": "datetime",
            }

            actual_schema, expected_schema = _relax_dtype(
                RELAX_CONVERSION, actual_schema, expected_schema
            )

    elif isinstance(df, pyspark.sql.DataFrame):
        # pylint: disable=unnecessary-comprehension
        actual_schema = {col: dtype for col, dtype in df.dtypes}
        # spark might inconsistently cast between int and bigint
        # date written to csv and re-read back might be timestamp
        if relax:
            RELAX_CONVERSION = {  # pylint: disable=invalid-name
                "tinyint": "numeric",
                "smallint": "numeric",
                "int": "numeric",
                "long": "numeric",
                "bigint": "numeric",
                "string": "string",
                "float": "double",
                "double": "double",
                "date": "timestamp",
                "timestamp": "timestamp",
                "array<int>": "array<numeric>",
                "array<long>": "array<numeric>",
                "array<bigint>": "array<numeric>",
            }

            actual_schema, expected_schema = _relax_dtype(
                RELAX_CONVERSION, actual_schema, expected_schema
            )

    else:
        if raise_exc:
            raise ValueError(f"{df} should be a spark or pandas dataframe.")

    # check for allow subset
    if not allow_subset and raise_exc:
        assert set(list(expected_schema.keys())) == set(list(actual_schema.keys())), (
            f"Actual vs expected columns do not match up. "
            f"Expected schema: {expected_schema}, actual schema: {actual_schema}"
        )

    # check for columns and dtypes
    in_expected_but_not_in_actual = set(expected_schema.items()) - set(
        actual_schema.items()
    )

    # raise exc for non-matching
    if in_expected_but_not_in_actual:
        # Convert schemas to dataframes
        df_expected = pd.DataFrame(
            list(expected_schema.items()), columns=["column_name", "data_type"]
        )
        df_actual = pd.DataFrame(
            list(actual_schema.items()), columns=["column_name", "data_type"]
        )

        # Merge the dataframes
        merged_df = df_expected.merge(
            df_actual, on=["column_name", "data_type"], how="outer", indicator=True
        )

        # Identify differences
        differences_df = merged_df[merged_df["_merge"] != "both"]

        # Check for differences and raise exception or log warning
        if not differences_df.empty:
            # Expected schema table
            expected_df = merged_df[merged_df["_merge"].isin(["left_only"])].drop(
                columns="_merge"
            )

            # Actual schema table
            actual_df = merged_df[merged_df["_merge"].isin(["right_only"])].drop(
                columns="_merge"
            )

            error_message = f"Schema differences found while checking {df_name}:\n\n\
            The following are the mismatched fields.\n\n\
                Expected schema:\n{expected_df.to_string(index=False)}\n\n\
                Actual schema:\n{actual_df.to_string(index=False)}"

            logger.warning(error_message)

            if raise_exc:
                raise ValueError(error_message)
