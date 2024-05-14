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
"""Tests."""

import pytest

from refit.v1.core.make_list_regexable import make_list_regexable


class TestMakeListRegexableWithPandas:
    def test_make_list_regexable(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df, params_keep_cols=[".*col"], enable_regex=True,
        )
        assert len(result_df.columns) == len(pandas_df.columns)

    def test_make_list_regexable_with_explicit_names(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df,
            params_keep_cols=["int_col", "float_col", "string_col"],
            enable_regex=True,
        )
        assert len(result_df.columns) == len(pandas_df.columns)

    def test_make_list_regexable_with_combination(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df,
            params_keep_cols=["int_.*", "float_col", "string_col"],
            enable_regex=True,
        )
        assert len(result_df.columns) == len(pandas_df.columns)


class TestMakeListRegexableWithSpark:
    def test_make_list_regexable(self, spark_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df.select(*params_keep_cols)
            return result_df

        result_df = accept_regexable_list(
            df=spark_df, params_keep_cols=[".*col"], enable_regex=True
        )
        assert len(result_df.columns) == len(spark_df.columns)

    def test_make_list_regexable_with_explicit_names(self, spark_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df.select(*params_keep_cols)
            return result_df

        result_df = accept_regexable_list(
            df=spark_df,
            params_keep_cols=["int_col", "float_col", "string_col"],
            enable_regex=True,
        )
        assert len(result_df.columns) == len(spark_df.columns)

    def test_make_list_regexable_with_combination(self, spark_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df.select(*params_keep_cols)
            return result_df

        result_df = accept_regexable_list(
            df=spark_df,
            params_keep_cols=["int_.*", "float_col", "string_col"],
            enable_regex=True,
        )
        assert len(result_df.columns) == len(spark_df.columns)


class TestMakeListRegexableFundamentalChecks:
    def test_raise_exc_default(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df, params_keep_cols=["ftr1_.*"], enable_regex=True
        )
        assert len(result_df.columns) == 0

    def test_raise_exc_enabled(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols", raise_exc=True,
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        with pytest.raises(
            ValueError, match=f"The following regex did not return a result: ftr1_.*.",
        ):
            accept_regexable_list(pandas_df, ["ftr1_.*"], enable_regex=True)

    def test_make_list_regexable_accept_args(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df, params_keep_cols=[".*col"], enable_regex=True
        )
        assert len(result_df.columns) == len(pandas_df.columns)

    def test_make_list_regexable_not_present(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, enable_regex):
            result_df = df
            return result_df

        result_df = accept_regexable_list(df=pandas_df, enable_regex=True)
        assert len(result_df.columns) == len(pandas_df.columns)

    def test_make_list_regexable_empty(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df
            return result_df

        result_df = accept_regexable_list(
            df=pandas_df, params_keep_cols=[], enable_regex=True
        )
        assert len(result_df.columns) == len(pandas_df.columns)

    def test_make_list_regexable_source_df_not_present(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(params_keep_cols, enable_regex):
            params_keep_cols = params_keep_cols
            return params_keep_cols

        with pytest.raises(
            ValueError, match="Please provide source dataframe",
        ):
            accept_regexable_list(params_keep_cols=["col.*"], enable_regex=True)

    def test_make_list_regexable_with_wrong_input_type(self, pandas_df):
        @make_list_regexable(
            source_df="df", make_regexable="params_keep_cols",
        )
        def accept_regexable_list(df, params_keep_cols, enable_regex):
            result_df = df[params_keep_cols]
            return result_df

        with pytest.raises(
            TypeError, match="'int' object is not iterable",
        ):
            accept_regexable_list(df=pandas_df, params_keep_cols=7, enable_regex=True)
