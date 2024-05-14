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
"""Contains tests."""


import pandas as pd
import pyspark.sql.functions as f
import pytest
from sklearn.impute import SimpleImputer

from refit.tests.v1.conftest import CustomException, dummy_func
from refit.v1.core.augment import augment


@augment()
def my_func_pd(*args, **kwargs):
    def dummy_func(df, x, y):
        df["new"] = df["c1"] + x + y
        return df

    return dummy_func(*args, **kwargs)


@augment()
def my_func_pd_imputer(*args, **kwargs):
    def dummy_func(df, x, y, imputer):
        df["new"] = df["c1"] + x + y
        return df, imputer

    return dummy_func(*args, **kwargs)


@augment()
def my_func_spark(*args, **kwargs):
    def dummy_func(df, x, y):
        df = df.withColumn("new", f.lit(x) + f.lit(y) + 1)
        return df

    return dummy_func(*args, **kwargs)


@augment()
def dummy_func_raises(*args, **kwargs):
    def dummy_raises(x):
        raise TypeError("TypeError")

    return dummy_raises(*args, **kwargs)


@augment()
def dummy_func_raises_custom_exception(*args, **kwargs):
    def dummy_raise_custom_exception(x):
        raise CustomException(x)

    return dummy_raise_custom_exception(*args, **kwargs)


@augment()
def dummy_func_no_exceptions(*args, **kwargs):
    return dummy_func(*args, **kwargs)


class TestNodeUnpack:
    def test_augment_pd_unpack_with_unpack(self, dummy_pd_df):
        result = my_func_pd(**{"df": dummy_pd_df, "x": 1, "y": 0})
        assert result["new"].tolist() == [2]

    def test_augment_pd_unpack_without_unpacking(self, dummy_pd_df):
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            my_func_pd({"df": dummy_pd_df, "x": 1, "y": 0})

    def test_augment_pd_unpack_without_bad_kw(self, dummy_pd_df):
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            my_func_pd({"df": dummy_pd_df, "x": 1, "y": 0, "unpak": True})

    def test_augment_pd_unpack_with_arg_unpack(self, dummy_pd_df):
        result = my_func_pd(unpack={"df": dummy_pd_df, "x": 1, "y": 0,})
        assert result["new"].tolist() == [2]

    def test_augment_pd_unpack3(self, dummy_pd_df):
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            my_func_pd({"df": dummy_pd_df, "x": 1, "y": 0})

    def test_augment_spark_unpack1(self, dummy_spark_df):
        result = my_func_spark(df=dummy_spark_df, x=1, y=1)
        assert [x.asDict() for x in result.select("new").collect()] == [{"new": 3}]

    def test_augment_spark_unpack2(self, dummy_spark_df):
        result = my_func_spark(df=dummy_spark_df, **{"x": 1, "y": 1})
        assert [x.asDict() for x in result.select("new").collect()] == [{"new": 3}]


class TestNodeInjectObject:
    def test_inject(self, dummy_pd_df):
        df, imputer = my_func_pd_imputer(
            **{
                "df": dummy_pd_df,
                "x": 1,
                "y": 0,
                "imputer": {"object": "sklearn.impute.SimpleImputer"},
            }
        )
        assert isinstance(imputer, SimpleImputer)

        assert df["new"].tolist() == [2]

    def test_inject_unpack(self, dummy_pd_df):
        df, imputer = my_func_pd_imputer(
            unpack={
                "df": dummy_pd_df,
                "x": 1,
                "y": 0,
                "imputer": {"object": "sklearn.impute.SimpleImputer"},
            }
        )
        assert isinstance(imputer, SimpleImputer)
        assert df["new"].tolist() == [2]


class TestRetry:
    def test_no_error(self):
        result = dummy_func_no_exceptions(
            **{
                "x": 1,
                "retry": {
                    "exception": {
                        "object": "builtins.TypeError",
                        "instantiate": False,
                    },
                    "max_tries": 5,
                    "interval": 0.01,
                },
            }
        )
        assert result == 1

    def test_no_retry(self):
        with pytest.raises(TypeError):
            dummy_func_raises(x=1)

    def test_with_retry(self, caplog):
        try:
            dummy_func_raises(
                **{
                    "x": 1,
                    "retry": {
                        "exception": {
                            "object": "builtins.TypeError",
                            "instantiate": False,
                        },
                        "max_tries": 5,
                        "interval": 0.01,
                    },
                }
            )
        except:
            pass

        assert len(caplog.record_tuples) == 5

    def test_with_retry_but_not_covered(self):
        with pytest.raises(TypeError):
            dummy_func_raises(
                **{
                    "x": 1,
                    "retry": {
                        "exception": {
                            "object": "builtins.ValueError",
                            "instantiate": False,
                        },
                        "max_tries": 5,
                        "interval": 0.01,
                    },
                }
            )

    def test_with_retry_custom_exception(self, caplog):
        try:
            dummy_func_raises_custom_exception(
                **{
                    "x": 1,
                    "retry": {
                        "exception": {
                            "object": "refit.tests.v1.conftest.CustomException",
                            "instantiate": False,
                        },
                        "max_tries": 5,
                        "interval": 0.01,
                    },
                }
            )
        except:
            pass

        assert len(caplog.record_tuples) == 5


class TestInputKwargFilterPandas:
    def test_input_kwarg_filter_pandas(self, dummy_pd_df):
        result = my_func_pd(
            df=dummy_pd_df, x=1, y=1, **{"kwarg_filter": {"df": "c1 == 1"}}
        )
        assert result.shape[0] == 1

    def test_input_kwarg_filter_pandas_python(self, sample_df_pd_all_dtypes):
        result = dummy_func_no_exceptions(
            x=sample_df_pd_all_dtypes,
            kwarg_filter={"x": "string_col.str.contains('fo')", "engine": "python"},
        )
        assert result.shape[0] == 1


class TestInputKwargFilterSpark:
    def test_input_kwarg_filter_spark(self, dummy_spark_df):
        result = my_func_spark(
            df=dummy_spark_df, x=1, y=1, **{"kwarg_filter": {"df": "c1 = 0"}}
        )
        assert result.count() == 0


class TestInputKwargSelectPandas:
    def test_input_kwarg_select_pandas(self, dummy_pd_df):
        with pytest.raises(KeyError):
            result = my_func_pd(
                df=dummy_pd_df, x=1, y=1, **{"kwarg_select": {"df": ["c1 as test"]}}
            )


class TestInputKwargSelectSpark:
    def test_input_kwarg_select_spark(self, dummy_spark_df):
        result = my_func_spark(
            df=dummy_spark_df, x=1, y=1, **{"kwarg_select": {"df": ["c1 as test"]}}
        )
        assert result.schema.names == ["test", "new"]


@augment()
def dummy_func_has_schema(*args, **kwargs):
    return dummy_func(*args, **kwargs)


def dummy_func_list(x, y):
    """Dummy function for testing purposes."""
    return x, y


@augment()
def dummy_func_has_schema_list(*args, **kwargs):
    return dummy_func_list(*args, **kwargs)


class TestInputHasSchema:
    def test_input_has_schema(self, dummy_pd_df):
        result = dummy_func_has_schema(
            x=dummy_pd_df,
            input_has_schema={
                "df": "x",
                "expected_schema": {"c1": "int"},
                "allow_subset": False,
                "raise_exc": False,
                "relax": False,
            },
        )
        assert dummy_pd_df.equals(result)

    def test_input_has_schema_list(self, dummy_pd_df):
        result_x, result_y = dummy_func_has_schema_list(
            x=dummy_pd_df,
            y=dummy_pd_df,
            input_has_schema=[
                {
                    "df": "x",
                    "expected_schema": {"c1": "int"},
                    "allow_subset": False,
                    "raise_exc": False,
                    "relax": False,
                },
                {
                    "df": "y",
                    "expected_schema": {"c1": "int"},
                    "allow_subset": False,
                    "raise_exc": False,
                    "relax": False,
                },
            ],
        )
        assert dummy_pd_df.equals(result_x)
        assert dummy_pd_df.equals(result_y)


class TestOutputHasSchema:
    def test_output_has_schema(self, dummy_pd_df):
        result = dummy_func_has_schema(
            x=dummy_pd_df,
            output_has_schema={
                "expected_schema": {"c1": "int"},
                "allow_subset": False,
                "raise_exc": False,
                "relax": True,
            },
        )
        assert dummy_pd_df.equals(result)

    def test_output_has_schema_list(self, dummy_pd_df):
        result_x, result_y = dummy_func_has_schema_list(
            x=dummy_pd_df,
            y=dummy_pd_df,
            output_has_schema=[
                {
                    "expected_schema": {"c1": "int"},
                    "allow_subset": False,
                    "raise_exc": False,
                    "relax": False,
                },
                {
                    "expected_schema": {"c1": "int"},
                    "allow_subset": False,
                    "raise_exc": False,
                    "relax": False,
                },
            ],
        )
        assert dummy_pd_df.equals(result_x)
        assert dummy_pd_df.equals(result_y)

    def test_invalid_input_schema(self, dummy_pd_df):
        with pytest.raises(ValueError, match="Schema differences found"):
            dummy_func_has_schema(
                x=dummy_pd_df,
                input_has_schema={
                    "df": "x",
                    "expected_schema": {"c1": "string"},
                    "allow_subset": False,
                    "raise_exc": True,
                    "relax": False,
                },
            )

    def test_invalid_output_schema(self, dummy_pd_df):
        with pytest.raises(ValueError, match="Schema differences found"):
            dummy_func_has_schema(
                x=dummy_pd_df,
                output_has_schema={
                    "expected_schema": {"c1": "string"},
                    "allow_subset": False,
                    "raise_exc": True,
                    "relax": False,
                },
            )


@augment()
def dummy_func_fill_na(*args, **kwargs):
    return dummy_func(*args, **kwargs)


class TestFillNa:
    def test_fillna(self, pandas_nulls_df):
        result = dummy_func_fill_na(
            pandas_nulls_df, fill_nulls={"value": False, "column_list": ["bool_col"]},
        )
        expected_data = [
            (1, None, "foo", True),
            (2, 2.0, None, False),
            (3, 2.0, "sample", False),
            (None, None, "awesome", True),
        ]

        expected = pd.DataFrame(
            expected_data, columns=["int_col", "float_col", "string_col", "bool_col"]
        )
        assert expected.equals(result)
