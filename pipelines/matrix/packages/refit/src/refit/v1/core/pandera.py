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

"""Decorator to handle schema/data validation."""
import functools
import json
import logging
from typing import Dict, List, Union

import pandas as pd
import pandera as pa
import pandera.pyspark as py
import pyspark
from pandera.errors import SchemaErrors

from refit.v1.internals import _get_param_from_arg_and_kwarg

INPUT_VALIDATION = "input_validation"
OUTPUT_VALIDATION = "output_validation"
RAISE_EXEC_ON_INPUT = "raise_exec_on_input"
RAISE_EXEC_ON_OUTPUT = "raise_exec_on_output"


def validate():
    """Validate the schema/data of `input` or `output` according to given schema."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            schema_errors = {}

            input_schema, args, kwargs = _get_param_from_arg_and_kwarg(
                INPUT_VALIDATION, *args, **kwargs
            )

            if input_schema:
                for df in input_schema.keys():
                    dataframe = kwargs[df]
                    schema = input_schema[df]
                    schema_errors[df] = _validate_schema(
                        dataframe=dataframe, schema=schema
                    )

                raise_exec_on_input, args, kwargs = _get_param_from_arg_and_kwarg(
                    RAISE_EXEC_ON_INPUT, *args, **kwargs
                )

                _generate_error_log(
                    schema=input_schema,
                    schema_errors=schema_errors,
                    io="Input",
                    raise_exec_on_io=raise_exec_on_input,
                )

            output_schema, args, kwargs = _get_param_from_arg_and_kwarg(
                OUTPUT_VALIDATION, *args, **kwargs
            )

            raise_exec_on_output, args, kwargs = _get_param_from_arg_and_kwarg(
                RAISE_EXEC_ON_OUTPUT, *args, **kwargs
            )

            out = func(*args, **kwargs)

            if output_schema:
                if isinstance(out, (list, tuple, dict)):
                    for df in output_schema.keys():
                        dataframe = out[df]
                        schema = output_schema[df]
                        schema_errors[df] = _validate_schema(
                            dataframe=dataframe, schema=schema
                        )
                else:
                    schema = output_schema["out"]
                    schema_errors["out"] = _validate_schema(
                        dataframe=out, schema=schema
                    )

                _generate_error_log(
                    schema=output_schema,
                    schema_errors=schema_errors,
                    io="Output",
                    raise_exec_on_io=raise_exec_on_output,
                )

            return out

        return wrapper

    return decorator


def _validate_schema(
    dataframe: Union[pyspark.sql.DataFrame, pd.DataFrame],
    schema: Union[pa.DataFrameSchema, py.DataFrameSchema],
) -> Dict[str, Union[Dict, List]]:
    if isinstance(dataframe, pd.DataFrame):
        try:
            errors = []
            validated_df = schema.validate(dataframe, lazy=True)
        except SchemaErrors as e:
            for error in e.schema_errors:
                errors.append(error.args[0])
        return errors
    else:
        validated_df = schema.validate(dataframe)
        errors = validated_df.pandera.errors
        return errors


def _generate_error_log(
    schema: Dict[str, Union[py.DataFrameSchema, pa.DataFrameSchema]],
    schema_errors: Dict[str, Union[Dict, List]],
    io: str,
    raise_exec_on_io: bool = False,
):
    output_error_log = []
    for df in schema.keys():
        if isinstance(schema_errors[df], dict):
            if schema_errors[df]:
                output_error_log.append(
                    f"""Error log for {df}:\n"""
                    + json.dumps(schema_errors[df], indent=4)
                    + "\n\n"
                )
            else:
                logging.debug(f"No schema or data mismatch found for {df}")
        else:
            if schema_errors[df]:
                output_error_log.append(
                    """Error log for {0}:\n {1}
                    """.format(  # pylint: disable=consider-using-f-string
                        df, "\n".join(schema_errors[df])
                    )
                )
            else:
                logging.debug(f"No schema or data mismatch found for {df}")

    error_log = "".join(output_error_log)

    if error_log:
        if raise_exec_on_io:
            raise Exception(f"{io} schema and data validation failed.\n {error_log}")
        logging.warning(f"{io} schema and data validation failed.\n {error_log}")
    else:
        logging.debug(f"All {io.lower()} schema and data validations passed")
