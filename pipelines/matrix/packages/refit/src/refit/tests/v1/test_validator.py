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

# pylint: skip-file
# flake8: noqa

import logging

import pandas as pd
import pandera as pa
import pandera.pyspark as py
import pyspark
import pyspark.sql.types as T
import pytest

from refit.v1.core.pandera import validate


@validate()
def node1_spark(input_df: pyspark.sql.DataFrame):
    return input_df


@validate()
def node2_spark(input_df: pyspark.sql.DataFrame, input_df2: pyspark.sql.DataFrame):
    return {"out1": input_df, "out2": input_df2}


class TestSparkValidator:
    def test_spark_schema_and_data_validation(self, spark, spark_df, caplog):
        input_validation = {
            "input_df": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(T.FloatType()),
                    "string_col": py.Column(T.IntegerType()),
                },
            ),
        }
        output_validation = {
            "out": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(T.FloatType(), py.Check.ge(10.0)),
                    "string_col": py.Column(T.StringType()),
                },
            ),
        }

        raise_exec_on_input = False
        raise_exec_on_output = False

        df = node1_spark(
            input_df=spark_df,
            input_validation=input_validation,
            output_validation=output_validation,
            raise_exec_on_input=raise_exec_on_input,
            raise_exec_on_output=raise_exec_on_output,
        )

        assert "Error log for input_df:" in caplog.text
        assert (
            "expected column 'string_col' to have type IntegerType(), got StringType()"
            in caplog.text
        )
        assert "Error log for out:" in caplog.text
        assert (
            "column 'float_col' with type FloatType() failed validation greater_than_or_equal_to(10.0)"
            in caplog.text
        )
        assert df.count() == 3
        assert df.columns == ["int_col", "float_col", "string_col"]

    @pytest.mark.skip(
        reason="unique values validation is not possible currently for spark dataframe"
    )
    def test_spark_primary_key_check(self, spark, spark_df_with_pk, caplog):
        assert 1 == 1

    @pytest.mark.skip(
        reason="unique values validation is not possible currently for spark dataframe"
    )
    def test_spark_composite_primary_key(self, spark, caplog):
        assert 1 == 1

    @pytest.mark.skip(
        reason="unique values validation is not possible currently for spark dataframe"
    )
    def test_spark_primary_key_with_nulls(self, spark, caplog):
        assert 1 == 1

    def test_multiple_data_validations(
        self, spark, spark_df, sample_df_spark_all_dtypes, caplog
    ):
        input_validation = {
            "input_df": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(
                        T.FloatType(), [py.Check.ge(0), py.Check.eq(90)]
                    ),
                    "string_col": py.Column(
                        T.StringType(),
                        [py.Check.str_startswith("."), py.Check.str_endswith("-")],
                    ),
                },
            ),
        }
        output_validation = {
            "out1": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(T.FloatType()),
                    "string_col": py.Column(T.StringType(), nullable=True),
                },
            ),
            "out2": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "long_col": py.Column(T.LongType()),
                    "string_col": py.Column(T.StringType()),
                    "float_col": py.Column(T.FloatType()),
                    "double_col": py.Column(T.DoubleType()),
                    "date_col": py.Column(T.DateType()),
                    "datetime_col": py.Column(T.TimestampType()),
                    "array_int": py.Column(T.ArrayType(T.IntegerType())),
                }
            ),
        }

        raise_exec_on_input = False
        raise_exec_on_output = False

        with caplog.at_level(logging.DEBUG):
            df = node2_spark(
                input_df=spark_df,
                input_df2=sample_df_spark_all_dtypes,
                input_validation=input_validation,
                output_validation=output_validation,
                raise_exec_on_input=raise_exec_on_input,
                raise_exec_on_output=raise_exec_on_output,
            )

        assert "Error log for input_df" in caplog.text
        assert (
            "column 'float_col' with type FloatType() failed validation equal_to(90)"
            in caplog.text
        )
        assert (
            "column 'float_col' with type FloatType() failed validation greater_than_or_equal_to(90)"
            not in caplog.text
        )
        assert (
            "column 'string_col' with type StringType() failed validation str_startswith('.')"
            in caplog.text
        )
        assert (
            "column 'string_col' with type StringType() failed validation str_endswith('-')"
            in caplog.text
        )
        assert "All output schema and data validations passed" in caplog.text

    def test_spark_raise_exception_on_input(self, spark, spark_df):
        input_validation = {
            "input_df": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(T.FloatType()),
                    "string_col": py.Column(T.IntegerType()),
                },
            ),
        }

        raise_exec_on_input = True

        with pytest.raises(Exception, match="Input schema and data validation failed."):
            df = node1_spark(
                input_df=spark_df,
                input_validation=input_validation,
                raise_exec_on_input=raise_exec_on_input,
            )

    def test_spark_raise_exception_on_output(self, spark, spark_df):
        output_validation = {
            "out": py.DataFrameSchema(
                {
                    "int_col": py.Column(T.IntegerType()),
                    "float_col": py.Column(T.FloatType()),
                    "string_col": py.Column(T.IntegerType()),
                },
            ),
        }

        raise_exec_on_output = True

        with pytest.raises(
            Exception, match="Output schema and data validation failed."
        ):
            df = node1_spark(
                input_df=spark_df,
                output_validation=output_validation,
                raise_exec_on_output=raise_exec_on_output,
            )


@validate()
def node1_pandas(input_df: pd.DataFrame):
    return input_df


@validate()
def node2_pandas(df1: pd.DataFrame, df2: pd.DataFrame):
    return df1.drop_duplicates("id1"), df2


class TestPandasValidator:
    def test_pandas_schema_and_data_validation(self, pandas_df, caplog):
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {
                    "float_col": pa.Column(float, pa.Check.ge(10.0)),
                    "int_col": pa.Column(str),
                    "string_col": pa.Column(str),
                }
            )
        }
        raise_exec_on_input = False

        df = node1_pandas(
            input_df=pandas_df,
            input_validation=input_validation,
            raise_exec_on_input=raise_exec_on_input,
        )

        assert "Error log for input_df:" in caplog.text
        assert "expected series 'int_col' to have type str" in caplog.text
        assert "non-nullable series 'string_col' contains null values" in caplog.text
        assert (
            "failed element-wise validator number 0: greater_than_or_equal_to(10.0)"
            in caplog.text
        )
        assert len(df) == 3

    def test_pandas_primary_key_check(self, pandas_df_with_pk, caplog):
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {
                    "id1": pa.Column(int, unique=True, nullable=False),  # primary key
                    "id2": pa.Column(str, nullable=True),
                    "name": pa.Column(str, nullable=True),
                }
            )
        }
        output_validation = {
            "out": pa.DataFrameSchema(
                {"id1": pa.Column(str), "id2": pa.Column(str), "name": pa.Column(str)}
            )
        }

        raise_exec_on_input = False
        raise_exec_on_output = False

        df = node1_pandas(
            input_df=pandas_df_with_pk,
            input_validation=input_validation,
            output_validation=output_validation,
            raise_exec_on_input=raise_exec_on_input,
            raise_exec_on_output=raise_exec_on_output,
        )

        assert len(df) == 5
        assert "series 'id1' contains duplicate values" in caplog.text
        assert "Error log for out" in caplog.text

    def test_pandas_composite_primary_key(self, pandas_df_with_pk, pandas_df, caplog):
        input_validation = {
            "df1": pa.DataFrameSchema(
                {
                    "id1": pa.Column(int, nullable=False),  # composite primary key
                    "id2": pa.Column(str, nullable=False),  # composite primary key
                    "name": pa.Column(str, nullable=True),
                },
                unique=["id1", "id2"],  # checks joint uniqueness
            ),
            "df2": pa.DataFrameSchema(
                {".*_col": pa.Column(nullable=False, regex=True),}
            ),
        }

        output_validation = {
            0: pa.DataFrameSchema(
                {
                    "id1": pa.Column(int),  # composite primary key
                    "id2": pa.Column(str),  # composite primary key
                    "name": pa.Column(str, nullable=True),
                },
                unique=["id1", "id2"],
            ),
            1: pa.DataFrameSchema({".*_col": pa.Column(nullable=True, regex=True)}),
        }

        out = node2_pandas(
            df1=pandas_df_with_pk,
            df2=pandas_df,
            input_validation=input_validation,
            output_validation=output_validation,
        )

        assert len(pandas_df_with_pk) == 5
        assert len(out[0]) == 4
        assert "Error log for df1:\n columns '('id1', 'id2')' not unique" in caplog.text
        assert "non-nullable series 'string_col' contains null values:" in caplog.text
        assert (
            "Error log for 0: \ncolumns '('id1', 'id2')' not unique" not in caplog.text
        )
        assert "Error log for 1:" not in caplog.text

    def test_pandas_primary_key_with_nulls(self, pandas_pk_null_df, caplog):
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {"id.*": pa.Column(nullable=True, regex=True),}, unique=["id1", "id2"],
            )
        }

        raise_exec_on_input = False

        with caplog.at_level(logging.DEBUG):
            out = node1_pandas(
                input_df=pandas_pk_null_df,
                input_validation=input_validation,
                raise_exec_on_input=raise_exec_on_input,
            )

        assert len(out) == 5
        assert "All input schema and data validations passed" in caplog.text

    def test_pandas_multiple_data_validation(self, pandas_string_df, caplog):
        continents = ["Asia", "Africa", "Europe", "Antarctica"]
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {
                    "string_col": pa.Column(
                        str,
                        [
                            pa.Check.str_matches(r"^[A-Z]"),
                            pa.Check.isin(continents),
                            pa.Check(lambda x: len(x) < 20),
                        ],
                    )
                }
            )
        }

        with caplog.at_level(logging.DEBUG):
            out = node1_pandas(
                input_df=pandas_string_df, input_validation=input_validation
            )

        assert "All input schema and data validations passed" in caplog.text
        assert out.shape == (3, 1)

    def test_pandas_raise_exception_on_input(self, pandas_string_df):
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {"string_col": pa.Column(str, pa.Check.str_matches(r"^[a-z]"))}
            )
        }

        raise_exec_on_input = True

        with pytest.raises(Exception, match="Input schema and data validation failed."):
            out = node1_pandas(
                input_df=pandas_string_df,
                input_validation=input_validation,
                raise_exec_on_input=raise_exec_on_input,
            )

    def test_pandas_raise_exception_on_output(self, pandas_string_df):
        input_validation = {
            "input_df": pa.DataFrameSchema(
                {"string_col": pa.Column(str, pa.Check.str_matches(r"^[A-Z]"))}
            )
        }

        output_validation = {
            "out": pa.DataFrameSchema({"string_col": pa.Column(float)})
        }

        raise_exec_on_output = True

        with pytest.raises(
            Exception, match="Output schema and data validation failed."
        ):
            out = node1_pandas(
                input_df=pandas_string_df,
                input_validation=input_validation,
                output_validation=output_validation,
                raise_exec_on_output=raise_exec_on_output,
            )
