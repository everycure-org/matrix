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

import pyspark.sql.functions as f
import pytest

from refit.v1.core.inline_primary_key import primary_key

# ----- primary key validation by passing input and output dataframe -----


@primary_key(
    output=0, df="prm_x", primary_key=["int_col"], nullable=False,
)
def node_func_return_list(prm_x):
    df_new = prm_x
    return [df_new, "abc"]


def test_inline_primary_key_with_output_and_input(spark_df):
    with pytest.raises(ValueError, match=r"Please supply either df or output."):
        node_func_return_list(prm_x=spark_df)


# ----- primary key check for spark/pandas input dataframe without null check-----


class TestInputDataframePrimaryKeyWithoutNull:

    df_name = "prm_x"

    def test_inline_with_input_primary_key_without_null_without_dupe(
        self, spark_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col"],
        )
        def node_func_with_input_without_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_input_without_null_without_dupe(prm_x=spark_df)
        assert result_df.count() == spark_df.count()

    def test_inline_primary_key_with_input_without_null_with_dupe(
        self, spark_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["float_col"],
        )
        def node_func_with_input_without_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match=f"Primary key columns for index - {self.df_name}, \['float_col'\] has duplicate values.",
        ):
            node_func_with_input_without_null_with_dupe(prm_x=spark_df)

    def test_inline_primary_key_with_input_without_null_without_dupe_list1(
        self, spark_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col", "float_col"],
        )
        def node_func_with_input_without_null_without_dupe_list1(prm_x):
            return prm_x

        result = node_func_with_input_without_null_without_dupe_list1(prm_x=spark_df)
        assert result.count() == spark_df.count()

    def test_inline_primary_key_with_input_without_null_without_dupe_list2(
        self, spark_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col", "string_col"],
        )
        def node_func_with_input_without_null_without_dupe_list2(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match=f"Primary key columns for index - {self.df_name}, \['int_col', 'string_col'\] has null values.",
        ):
            node_func_with_input_without_null_without_dupe_list2(prm_x=spark_df)

    def test_inline_pandas_with_input_primary_key_without_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col"],
        )
        def node_func_with_input_without_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_input_without_null_without_dupe(prm_x=pandas_df)
        assert result_df.shape == pandas_df.shape

    def test_inline_pandas_primary_key_with_input_without_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["float_col"],
        )
        def node_func_with_input_without_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match=f"Primary key columns for index - {self.df_name}, \['float_col'\] has duplicate values.",
        ):
            node_func_with_input_without_null_with_dupe(prm_x=pandas_df)

    def test_inline_pandas_primary_key_with_input_without_null_without_dupe_list1(
        self, pandas_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col", "float_col"],
        )
        def node_func_with_input_without_null_without_dupe_list1(prm_x):
            return prm_x

        result = node_func_with_input_without_null_without_dupe_list1(prm_x=pandas_df)
        assert result.shape == pandas_df.shape


# ----- primary key check for spark/pandas input dataframe without null check-----


class TestInputDataframePrimaryKeyWithNull:

    df_name = "prm_x"

    def test_inline_primary_key_with_input_with_null_without_dupe(
        self, spark_df,
    ):
        @primary_key(df=self.df_name, primary_key=["int_col"], nullable=True)
        def node_func_with_input_with_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_input_with_null_without_dupe(prm_x=spark_df)
        assert result_df.count() == spark_df.count()

    def test_inline_primary_key_with_input_with_null_without_dupe_list(
        self, spark_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col", "string_col"], nullable=True
        )
        def node_func_with_input_with_null_without_dupe_list(prm_x):
            return prm_x

        result_df = node_func_with_input_with_null_without_dupe_list(prm_x=spark_df)
        assert result_df.count() == spark_df.count()

    def test_inline_primary_key_with_input_with_null_with_dupe(
        self, spark_df,
    ):
        @primary_key(df=self.df_name, primary_key=["float_col"], nullable=True)
        def node_func_with_input_with_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match=f"Primary key columns for index - {self.df_name}, \['float_col'\] has duplicate values.",
        ):
            node_func_with_input_with_null_with_dupe(prm_x=spark_df)

    def test_inline_pandas_primary_key_with_input_with_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(df=self.df_name, primary_key=["int_col"], nullable=True)
        def node_func_with_input_with_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_input_with_null_without_dupe(prm_x=pandas_df)
        assert result_df.shape == pandas_df.shape

    def test_inline_pandas_primary_key_with_input_with_null_without_dupe_list(
        self, pandas_df,
    ):
        @primary_key(
            df=self.df_name, primary_key=["int_col", "float_col"], nullable=True
        )
        def node_func_with_input_with_null_without_dupe_list1(prm_x):
            return prm_x

        result = node_func_with_input_with_null_without_dupe_list1(prm_x=pandas_df)
        assert result.shape == pandas_df.shape

    def test_inline_pandas_primary_key_with_input_with_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(df=self.df_name, primary_key=["float_col"], nullable=True)
        def node_func_with_input_with_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match=f"Primary key columns for index - {self.df_name}, \['float_col'\] has duplicate values.",
        ):
            node_func_with_input_with_null_with_dupe(prm_x=pandas_df)


# ----- primary key check for spark/pandas output dataframe without null check-----


class TestOutputDataframePrimaryKeyWithoutNull:
    def test_inline_primary_key_with_output_without_null_without_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["int_col"], output=0)
        def node_func_with_output_without_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_output_without_null_without_dupe(prm_x=spark_df)
        assert result_df.count() == spark_df.count()

    def test_inline_primary_key_with_output_without_null_with_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["float_col"], output=0)
        def node_func_with_output_without_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
        ):
            node_func_with_output_without_null_with_dupe(prm_x=spark_df)

    def test_inline_primary_key_multiple_output_without_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["int_col"], output=1)
        def node_func_full_example_multiple_output_without_dupe(df1, df2):
            df1_new = df1.withColumn("new_output1", f.lit(1))
            df2_new = df2.withColumn("new_output2", f.lit(1))
            return [df1_new, df2_new]

        node_func_full_example_multiple_output_without_dupe(df1=spark_df, df2=spark_df)

    def test_inline_primary_key_multiple_output_with_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["new_output2"], output=1)
        def node_func_full_example_multiple_output_with_dupe(df1, df2):
            df1_new = df1.withColumn("new_output1", f.lit(1))
            df2_new = df2.withColumn("new_output2", f.lit(1))
            return [df1_new, df2_new]

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 1, \['new_output2'\] has duplicate values.",
        ):
            node_func_full_example_multiple_output_with_dupe(df1=spark_df, df2=spark_df)

    def test_inline_pandas_primary_key_with_output_without_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["int_col"], output=0)
        def node_func_with_output_without_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_output_without_null_without_dupe(prm_x=pandas_df)
        assert result_df.shape == pandas_df.shape

    def test_inline_pandas_primary_key_with_output_without_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["float_col"], output=0)
        def node_func_with_output_without_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
        ):
            node_func_with_output_without_null_with_dupe(prm_x=pandas_df)

    def test_inline_pandas_primary_key_multiple_output_without_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["int_col"], output=1)
        def node_func_pandas_full_example_multiple_output_without_null_without_dupe(
            df1, df2
        ):
            df1["new_output1"] = 1
            df2["new_output2"] = 1
            return [df1, df2]

        node_func_pandas_full_example_multiple_output_without_null_without_dupe(
            df1=pandas_df, df2=pandas_df
        )

    def test_inline_pandas_primary_key_multiple_output_without_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["new_output2"], output=1)
        def node_func_pandas_full_example_multiple_output_without_null_with_dupe(
            df1, df2
        ):
            df1["new_output1"] = 1
            df2["new_output2"] = 1
            return [df1, df2]

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 1, \['new_output2'\] has duplicate values.",
        ):
            node_func_pandas_full_example_multiple_output_without_null_with_dupe(
                df1=pandas_df, df2=pandas_df
            )


# ----- primary key check for spark/pandas output dataframe with null check-----


class TestOutputDataframePrimaryKeyWithNull:
    def test_inline_primary_key_with_output_with_null_without_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["int_col"], output=0)
        def node_func_with_output_without_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_output_without_null_without_dupe(prm_x=spark_df)
        assert result_df.count() == spark_df.count()

    def test_inline_primary_key_with_output_with_null_with_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["float_col"], output=0)
        def node_func_with_output_without_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
        ):
            node_func_with_output_without_null_with_dupe(prm_x=spark_df)

    def test_inline_primary_key_multiple_output_with_null_without_dupe(
        self, spark_df,
    ):
        @primary_key(primary_key=["int_col"], output=1)
        def node_func_full_example_multiple_output_without_dupe(df1, df2):
            df1_new = df1.withColumn("new_output1", f.lit(1))
            df2_new = df2.withColumn("new_output2", f.lit(1))
            return [df1_new, df2_new]

        node_func_full_example_multiple_output_without_dupe(df1=spark_df, df2=spark_df)

    def test_inline_primary_key_multiple_output_with_null_without_dupe2(
        self, spark_df,
    ):
        @primary_key(primary_key=["new_output2"], output=1)
        def node_func_full_example_multiple_output_with_dupe(df1, df2):
            df1_new = df1.withColumn("new_output1", f.lit(1))
            df2_new = df2.withColumn("new_output2", f.lit(1))
            return [df1_new, df2_new]

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 1, \['new_output2'\] has duplicate values.",
        ):
            node_func_full_example_multiple_output_with_dupe(df1=spark_df, df2=spark_df)

    def test_inline_pandas_primary_key_with_output_with_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["int_col"], output=0, nullable=True)
        def node_func_with_output_with_null_without_dupe(prm_x):
            return prm_x

        result_df = node_func_with_output_with_null_without_dupe(prm_x=pandas_df)
        assert result_df.shape == pandas_df.shape

    def test_inline_pandas_primary_key_with_output_with_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["float_col"], output=0, nullable=True)
        def node_func_with_output_with_null_with_dupe(prm_x):
            return prm_x

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 0, \['float_col'\] has duplicate values.",
        ):
            node_func_with_output_with_null_with_dupe(prm_x=pandas_df)

    def test_inline_pandas_primary_key_multiple_output_with_null_without_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["int_col"], output=1, nullable=True)
        def node_func_pandas_full_example_multiple_output_with_null_without_dupe(
            df1, df2
        ):
            df1["new_output1"] = 1
            df2["new_output2"] = 1
            return [df1, df2]

        node_func_pandas_full_example_multiple_output_with_null_without_dupe(
            df1=pandas_df, df2=pandas_df
        )

    def test_inline_pandas_primary_key_multiple_output_with_null_with_dupe(
        self, pandas_df,
    ):
        @primary_key(primary_key=["new_output2"], output=1, nullable=True)
        def node_func_pandas_full_example_multiple_output_with_null_with_dupe(df1, df2):
            df1["new_output1"] = 1
            df2["new_output2"] = 1
            return [df1, df2]

        with pytest.raises(
            TypeError,
            match="Primary key columns for index - 1, \['new_output2'\] has duplicate values.",
        ):
            node_func_pandas_full_example_multiple_output_with_null_with_dupe(
                df1=pandas_df, df2=pandas_df
            )
