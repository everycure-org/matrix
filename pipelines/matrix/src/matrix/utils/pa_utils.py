"""
Temp. wrapper module to enable data quality checks defined independently of the underlying
library used to represent the data. Currently supports Spark and Pandas dataframes.

NOTE: Should be removed as soon as Pandera supports this.
"""

import typing
from dataclasses import dataclass
from functools import wraps
from typing import ClassVar, Dict, List, Optional

import pandas as pd
import pandera as pa
import pandera.pyspark as psa
import pyspark.sql as ps
import pyspark.sql.types as T
from pandera.decorators import _handle_schema_error


@dataclass
class Type:
    """Class to represent a list of a given type."""

    type_: type
    is64: bool = False

    _type_map: ClassVar[Dict[type, type]] = {
        int: T.IntegerType,
        float: T.FloatType,
        str: T.StringType,
        bool: T.BooleanType,
    }

    _64_map: ClassVar[Dict[type, type]] = {
        int: T.LongType,
        float: T.DoubleType,
    }

    def build_for_type(self, type_):
        if type_ is pd.DataFrame:
            return self.type_

        if type_ is ps.DataFrame:
            if self.is64:
                return self._64_map[self.type_]()
            else:
                return self._type_map[self.type_]()

        raise NotImplementedError(f"generation for type {type_} currently not supported!")


@dataclass
class ArrayType(Type):
    """Class to represent an array of a given type."""

    def build_for_type(self, type_):
        if type_ is pd.DataFrame:
            return List[super().build_for_type(type_)]

        if type_ is ps.DataFrame:
            return T.ArrayType(super().build_for_type(type_))

        raise NotImplementedError(f"generation for type {type_} currently not supported!")


@dataclass
class Column:
    """Data class to represent a class agnostic Pandera Column."""

    type_: Type
    checks: Optional[List] = None
    nullable: bool = True

    def build_for_type(self, type_):
        if type_ is pd.DataFrame:
            return pa.Column(self.type_.build_for_type(type_), checks=self.checks, nullable=self.nullable)

        if type_ is ps.DataFrame:
            return psa.Column(self.type_.build_for_type(type_), checks=self.checks, nullable=self.nullable)

        raise NotImplementedError(f"generation for type {type_} currently not supported!")


@dataclass
class DataFrameSchema:
    """Data class to represent a class agnostic Pandera DataFrameSchema."""

    columns: Dict[str, Column]
    unique: Optional[List] = None

    _schema_map: ClassVar[Dict[type, type]] = {pd.DataFrame: pa.DataFrameSchema, ps.DataFrame: psa.DataFrameSchema}

    def build_for_type(self, type_) -> psa:
        return self._schema_map[type_](
            columns={name: col.build_for_type(type_) for name, col in self.columns.items()},
            unique=self.unique,
        )


def check_output(schema: DataFrameSchema, df_name: Optional[str] = None):
    """Decorator to validate output schema of decorated function.

    Args:
        schema: Schema to validate
        df_name (optional): name of output arg to validate
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
            output = func(*args, **kwargs)

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
