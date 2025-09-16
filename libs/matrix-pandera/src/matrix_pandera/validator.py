"""
Temp. wrapper module to enable data quality checks defined independently of the underlying
library used to represent the data. Currently supports Spark and Pandas dataframes.

NOTE: Should be removed as soon as Pandera supports this.
"""

import inspect
import typing
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional

import pandas as pd
import pandera as pa
import pandera.pyspark as pas
import pyspark.sql as ps
from pandera.decorators import _handle_schema_error


@dataclass
class Column:
    """Data class to represent a class agnostic Pandera Column."""

    type_: type
    checks: Optional[List] = None
    nullable: bool = True


@dataclass
class DataFrameSchema:
    """Data class to represent a class agnostic Pandera DataFrameSchema."""

    columns: Dict[str, Column]
    unique: Optional[List] = None
    strict: bool = False

    def build_for_type(self, type_) -> typing.Union[pas.DataFrameSchema, pa.DataFrameSchema]:
        # Build pandas version
        if type_ is pd.DataFrame:
            return pa.DataFrameSchema(
                columns={
                    name: pa.Column(col.type_, checks=col.checks, nullable=col.nullable)
                    for name, col in self.columns.items()
                },
                unique=self.unique,
                strict=self.strict,
            )

        # Build pyspark version
        if type_ is ps.DataFrame:
            return pas.DataFrameSchema(
                columns={
                    name: pas.Column(col.type_, checks=col.checks, nullable=col.nullable)
                    for name, col in self.columns.items()
                },
                unique=self.unique,
                strict=self.strict,
            )

        raise TypeError()


def check_input(schema: DataFrameSchema, df_name: str):
    """Decorator to validate output schema of decorated function.

    Args:
        schema: Schema to validate
        df_name (optional): name of output arg to validate
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get type hints for the function
            type_hints = typing.get_type_hints(func)

            # Bind args and kwargs to parameter names
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            value = bound.arguments.get(df_name)
            arg_type = type_hints.get(df_name)

            df_schema = schema.build_for_type(arg_type)

            if value is not None:
                try:
                    # breakpoint()
                    df_schema.validate(value, lazy=False)
                except pa.errors.SchemaError as e:
                    _handle_schema_error("check_input", func, df_schema, value, e)

            output = func(*args, **kwargs)

            return output

        return wrapper

    return decorator


def check_output(
    schema: DataFrameSchema, df_name: Optional[str] = None, raise_df_undefined: bool = True, pass_columns: bool = False
):
    """Decorator to validate output schema of decorated function.

    Args:
        schema: Schema to validate
        df_name (optional): name of output arg to validate
        raise_df_undefined: if set, ensures error is thrown if `df_name` is not defined
        pass_cols (optional): boolean indicating whether cols should be passed into callable
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (type_ := typing.get_type_hints(func).get("return")):
                raise TypeError("No output typehint specified!")

            if df_name:
                if not typing.get_origin(typing.Dict) == dict:
                    raise TypeError("Specified df_name arg, but function output typehint is not dict.")

                type_ = typing.get_args(type_)[1]

            df_schema = schema.build_for_type(type_)

            if pass_columns:
                output = func(*args, **kwargs, cols=list(schema.columns.keys()))
            else:
                output = func(*args, **kwargs)

            if df_name is not None:
                df = output.get(df_name)

                if df is None and raise_df_undefined:
                    raise RuntimeError(f"df {df_name} not found!")
            else:
                df = output

            if df is not None:
                try:
                    df_schema.validate(df, lazy=False)
                except pa.errors.SchemaError as e:
                    _handle_schema_error("check_output", func, df_schema, df, e)

            return output

        return wrapper

    return decorator
