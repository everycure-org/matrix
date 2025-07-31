from typing import Dict

import pandas as pd
import pandera as pa
import pyspark.sql as ps
import pytest
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output
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
        @check_output(DataFrameSchema(columns={"id": Column(int)}))
        def test_fn(pandas_df):
            return pandas_df

        # When invoking the function
        test_fn(pandas_df)

        # Then TypeError is thrown


@pytest.mark.parametrize(
    "schema",
    [
        # Schema validating int column
        DataFrameSchema(columns={"int": Column(int)}),
        # Schema validating array column
        DataFrameSchema(columns={"bool": Column(bool)}),
        # Schema validating array column
        DataFrameSchema(columns={"list": Column(list[float])}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(
            columns={"id": Column(int), "bool": Column(bool)},
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
    [DataFrameSchema(columns={"int": Column(int)})],
)
def test_check_output_supports_validating_mappings_to_frames(pandas_df, schema):
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
        DataFrameSchema(columns={"int": Column(float)}),
        # Float array not int array
        DataFrameSchema(columns={"list": Column(int)}),
        # Validata primary key
        DataFrameSchema(columns={}, unique=["int"]),
    ],
)
def test_pandas_dataframe_failures(pandas_df, schema):
    # Given a function that validates the output schema
    @check_output(schema)
    def to_test(df: pd.DataFrame) -> pd.DataFrame:
        return df

    with pytest.raises(pa.errors.SchemaError):
        # When invoking the test function
        to_test(pandas_df)

        # Then error returned


@pytest.mark.parametrize(
    "schema",
    [
        # Schema validating int column
        DataFrameSchema(columns={"int": Column(T.IntegerType())}),
        # Schema validating long column
        DataFrameSchema(columns={"long": Column(T.LongType)}),
        # Schema validating bool column
        DataFrameSchema(columns={"bool": Column(T.BooleanType())}),
        # Schema validating float array column
        DataFrameSchema(columns={"float_list": Column(T.ArrayType(T.FloatType()))}),
        # Schema validating int array column
        DataFrameSchema(columns={"double_list": Column(T.ArrayType(T.DoubleType()))}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(
            columns={
                "bool": Column(T.BooleanType()),
                "long": Column(T.LongType()),
                "float_list": Column(T.ArrayType(T.FloatType())),
                "double_list": Column(T.ArrayType(T.DoubleType())),
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
        DataFrameSchema(columns={"int": Column(T.LongType())}),
        # Long is not int
        DataFrameSchema(columns={"long": Column(T.IntegerType())}),
        # Float list is not long
        DataFrameSchema(columns={"float_list": Column(T.ArrayType(T.LongType()))}),
        # Long list is not float
        DataFrameSchema(columns={"double_list": Column(T.ArrayType(T.FloatType()))}),
        # Schema validating multiple columns and uniques
        DataFrameSchema(columns={}, unique=["int"]),
    ],
)
def test_pyspark_dataframe_errors(spark_df, schema):
    # Given a function that validates the output schema
    @check_output(schema)
    def to_test(df: ps.DataFrame) -> ps.DataFrame:
        return df

    with pytest.raises(pa.errors.SchemaError):
        # When invoking the test function
        to_test(spark_df)

        # Then error returned
