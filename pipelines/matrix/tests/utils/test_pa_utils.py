from typing import Dict

import pandas as pd
import pandera as pa
import pyspark.sql as ps
import pytest
from matrix.utils.pa_utils import ArrayType, Column, DataFrameSchema, Type, check_output
from pyspark.sql import types as T
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def pandas_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "int": [1, 1], "list": [[0.1, 0.2], [0.3, 0.4]], "bool": [True, False]})


@pytest.fixture
def spark_df(spark) -> ps.DataFrame:
    """Create sample edges dataframe."""
    return spark.createDataFrame(
        [(1, 1, 20000, [1.0, 2.0], [1000.0, 1000.0], True), (2, 1, 20000, [1.0, 3.0], [1000.0, 1000.0], False)],
        schema=T.StructType(
            [
                T.StructField("id", T.IntegerType()),
                T.StructField("int", T.IntegerType()),
                T.StructField("long", T.LongType()),
                T.StructField("float_list", T.ArrayType(T.FloatType())),
                T.StructField("double_list", T.ArrayType(T.DoubleType())),
                T.StructField("bool", T.BooleanType()),
            ]
        ),
    )


def test_function_without_typehint_throws_error(pandas_df):
    with pytest.raises(TypeError):
        # Given function without output typehint
        @check_output(DataFrameSchema(columns={"id": Column(Type(int))}))
        def test_fn(pandas_df):
            return pandas_df

        # When invoking the function
        test_fn(pandas_df)

        # Then TypeError is thrown


@pytest.mark.parametrize(
    "schema",
    [
        # Schema validating int column
        DataFrameSchema(columns={"int": Column(Type(int))}),
        # Schema validating array column
        DataFrameSchema(columns={"bool": Column(Type(bool))}),
        # Schema validating array column
        DataFrameSchema(columns={"list": Column(ArrayType(float))}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(
            columns={"id": Column(Type(int)), "bool": Column(Type(bool)), "list": Column(ArrayType(float))},
            unique=["id"],
        ),
    ],
)
def test_pandas_dataframe_valid(pandas_df, schema):
    # Given a function that validates the output schema
    @check_output(schema)
    def to_test(df: pd.DataFrame) -> pd.DataFrame:
        return df

    # When invoking the test function
    result = to_test(pandas_df)

    # Then initial dataframe returned
    pd.testing.assert_frame_equal(result, pandas_df)


@pytest.mark.parametrize(
    "schema",
    [DataFrameSchema(columns={"int": Column(Type(int))})],
)
def test_pandas_dataframe_dict_valid(pandas_df, schema):
    # Given a function that validates the output schema
    @check_output(schema, df_name="df")
    def to_test(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {"df": df}

    # When invoking the test function
    result = to_test(pandas_df)

    # Then initial dataframe returned
    pd.testing.assert_frame_equal(result["df"], pandas_df)


@pytest.mark.parametrize(
    "schema",
    [
        # Int column is not of type float
        DataFrameSchema(columns={"int": Column(Type(float))}),
        # Float array not int array
        DataFrameSchema(columns={"list": Column(ArrayType(int))}),
        # Validata primary key
        DataFrameSchema(columns={}, unique=["int"]),
    ],
)
def test_pandas_dataframe_failures(pandas_df, schema):
    with pytest.raises(pa.errors.SchemaError):
        # Given a function that validates the output schema
        @check_output(schema)
        def to_test(df: pd.DataFrame) -> pd.DataFrame:
            return df

        # When invoking the test function
        to_test(pandas_df)

        # Then error returned


@pytest.mark.parametrize(
    "schema",
    [
        # Schema validating int column
        DataFrameSchema(columns={"int": Column(Type(int))}),
        # Schema validating long column
        DataFrameSchema(columns={"long": Column(Type(int, is64=True))}),
        # Schema validating bool column
        DataFrameSchema(columns={"bool": Column(Type(bool))}),
        # Schema validating float array column
        DataFrameSchema(columns={"float_list": Column(ArrayType(float))}),
        # Schema validating int array column
        DataFrameSchema(columns={"double_list": Column(ArrayType(float, is64=True))}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(
            columns={
                "bool": Column(Type(bool)),
                "long": Column(Type(int, is64=True)),
                "float_list": Column(ArrayType(float)),
                "double_list": Column(ArrayType(float, is64=True)),
            },
            unique=["id"],
        ),
    ],
)
def test_pyspark_dataframe_valid(spark_df, schema):
    # Given a function that validates the output schema
    @check_output(schema)
    def to_test(df: ps.DataFrame) -> ps.DataFrame:
        return df

    # When invoking the test function
    result = to_test(spark_df)

    # Assert frames equal
    assertDataFrameEqual(result, spark_df)


@pytest.mark.parametrize(
    "schema",
    [
        # Int is not long
        DataFrameSchema(columns={"int": Column(Type(int, is64=True))}),
        # Long is not int
        DataFrameSchema(columns={"long": Column(Type(int))}),
        # Float list is not long
        DataFrameSchema(columns={"float_list": Column(ArrayType(float, is64=True))}),
        # Long list is not float
        DataFrameSchema(columns={"double_list": Column(ArrayType(float))}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(columns={}, unique=["int"]),
    ],
)
def test_pyspark_dataframe_errors(spark_df, schema):
    with pytest.raises(pa.errors.SchemaError):
        # Given a function that validates the output schema
        @check_output(schema)
        def to_test(df: ps.DataFrame) -> ps.DataFrame:
            return df

        # When invoking the test function
        to_test(spark_df)

        # Then error returned
