import typing
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd
import pandera as pa
import pandera.pyspark as psa
import pyspark.sql as ps
import pyspark.sql.types as T
from pandera.decorators import _handle_schema_error


@dataclass
class Column:
    """Data class to represent a class agnostic Pandera Column."""

    type_: Any
    checks: Optional[List] = None
    nullable: bool = True


@dataclass
class DataFrameSchema:
    """Data class to represent a class agnostic Pandera DataFrameSchema."""

    columns: Dict[str, Column]
    unique: Optional[List] = None

    def build_for_class(self, cls) -> psa:
        # Build pandas version
        if cls is pd.DataFrame:
            return pa.DataFrameSchema(
                columns={
                    name: pa.Column(col.type_, checks=col.checks, nullable=col.nullable)
                    for name, col in self.columns.items()
                },
                unique=self.unique,
            )

        # Build pyspark version
        if cls is ps.DataFrame:
            return psa.DataFrameSchema(
                columns={
                    name: psa.Column(col.type_, checks=col.checks, nullable=col.nullable)
                    for name, col in self.columns.items()
                },
                unique=self.unique,
            )

        raise TypeError()


def check_output(schema: DataFrameSchema):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract return type of function
            if not (type_ := typing.get_type_hints(func).get("return")):
                raise RuntimeError(f"No output typehint specified!")

            # Build validator
            df_schema: psa.DataFrameSchema = schema.build_for_class(type_)

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


@check_output(DataFrameSchema(columns={"id": Column(int)}))
def dummy_pandas_fn() -> pd.DataFrame:
    return pd.DataFrame({"id": [1]})


@check_output(DataFrameSchema(columns={"bucket": Column(int, nullable=False)}))
def dummy_spark_fn(num_buckets: int = 10) -> ps.DataFrame:
    # Construct df to bucketize
    spark_session: ps.SparkSession = ps.SparkSession.builder.getOrCreate()

    # Bucketize df
    return spark_session.createDataFrame(
        data=[(bucket,) for bucket in range(num_buckets)],
        schema=T.StructType([T.StructField("bucket", T.IntegerType())]),
    )


# Invoke the fn
dummy_spark_fn()
dummy_pandas_fn()
