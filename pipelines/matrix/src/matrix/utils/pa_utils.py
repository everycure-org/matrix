"""
Temp. wrapper module to enable data quality checks defined independently of the underlying
library used to represent the data. Currently supports Spark and Pandas dataframes.

NOTE: Should be removed as soon as Pandera supports this.
"""

import typing
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional

import pandas as pd
import pandera as pa
import pandera.pyspark as pas
import pyspark.sql as ps
import pyspark.sql.types as T
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


def check_output(schema: DataFrameSchema, df_name: Optional[str] = None, pass_columns: bool = False):
    """Decorator to validate output schema of decorated function.

    Args:
        schema: Schema to validate
        df_name (optional): name of output arg to validate
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
                output = func(schema.columns.keys(), *args, **kwargs)

            if df_name is not None:
                df = output[df_name]
            else:
                df = output

            try:
                df_schema.validate(df, lazy=False)
            except pa.errors.SchemaError as e:
                _handle_schema_error("check_output", func, df_schema, df, e)

            return output

        return wrapper

    return decorator
