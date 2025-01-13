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

    _type_map: ClassVar[Dict[type, type]] = {int: T.IntegerType, float: T.FloatType, str: T.StringType}

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


@dataclass
class ArrayType(Type):
    """Class to represent an array of a given type."""

    ...

    def build_for_type(self, type_):
        if type_ is pd.DataFrame:
            return List[super().build_for_type()]

        if type_ is ps.DataFrame:
            return T.ArrayType(super().build_for_type(type_))


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


def check_output(schema: DataFrameSchema):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract return type of function
            if not (type_ := typing.get_type_hints(func).get("return")):
                raise RuntimeError(f"No output typehint specified!")

            # Build validator
            df_schema: psa.DataFrameSchema = schema.build_for_type(type_)

            # Invoke function
            df = func(*args, **kwargs)
            # Run validation
            try:
                df_schema.validate(df, lazy=False)
            except pa.errors.SchemaError as e:
                _handle_schema_error("check_output", func, df_schema, df, e)

            return df

        return wrapper

    return decorator


@check_output(DataFrameSchema(columns={"id": Column(Type(int))}, unique=["id"]))
def dummy_pandas_fn() -> pd.DataFrame:
    return pd.DataFrame({"id": [1]})


@check_output(
    DataFrameSchema(
        columns={
            "bucket": Column(Type(int), nullable=False),
            "embedding": Column(ArrayType(int, is64=True), nullable=False),
        },
        unique=["bucket"],
    )
)
def dummy_spark_fn(num_buckets: int = 10) -> ps.DataFrame:
    # Construct df to bucketize
    spark_session: ps.SparkSession = ps.SparkSession.builder.getOrCreate()

    # Bucketize df
    return spark_session.createDataFrame(
        data=[(bucket, [1, 2, 3]) for bucket in range(num_buckets)],
        schema=T.StructType(
            [T.StructField("bucket", T.IntegerType()), T.StructField("embedding", T.ArrayType(T.LongType()))]
        ),
    )


# Invoke the fn
dummy_spark_fn()
dummy_pandas_fn()
