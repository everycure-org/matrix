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

import pandas as pd
import pyspark.sql.functions as f
import pytest

from refit.v1.core.inline_has_schema import has_schema


# ----- has_schema pyspark -----
@has_schema(
    output=0,
    df="prm_x",
    schema={"site_id": "int", "site_name": "string"},
    allow_subset=False,
)
def node_func_return_list(prm_x):
    df_new = prm_x
    return [df_new, "abc"]


def test_has_schema_allow_subset_false(sample_df_spark_all_dtypes):
    with pytest.raises(ValueError, match=r"Please supply only df or output."):
        node_func_return_list(prm_x=sample_df_spark_all_dtypes)


@has_schema(
    schema={"site_id": "int", "site_name": "string"}, allow_subset=False,
)
def node_func_with_new_col1(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return df_new


def test_default_to_output_allow_subset_false(sample_df_spark_all_dtypes):
    with pytest.raises(
        AssertionError, match=r"Actual vs expected columns do not match up."
    ):
        node_func_with_new_col1(df=sample_df_spark_all_dtypes)


@has_schema(
    schema={"new_output": "int"}, allow_subset=True,
)
def node_func_with_new_col2(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return df_new


def test_default_to_output_allow_subset_true(sample_df_spark_all_dtypes):
    node_func_with_new_col2(df=sample_df_spark_all_dtypes)


@has_schema(
    output=0, schema={"new_output": "int"}, allow_subset=True,
)
def node_func_return_list1(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_choice_0_is_df(sample_df_spark_all_dtypes):
    node_func_return_list1(df=sample_df_spark_all_dtypes)


@has_schema(
    output=1, schema={"new_output": "int"}, allow_subset=True,
)
def node_func_return_list2(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_choice_1_is_not_df(sample_df_spark_all_dtypes):
    with pytest.raises(ValueError, match=r"should be a spark or pandas dataframe."):
        node_func_return_list2(df=sample_df_spark_all_dtypes)


@has_schema(
    output=0, schema={"new_output": "int", "new_output2": "int"}, allow_subset=True,
)
def node_func_bad_schema(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_choice_0_bad_schema(sample_df_spark_all_dtypes):
    with pytest.raises(ValueError, match=r"Schema differences found"):
        node_func_bad_schema(df=sample_df_spark_all_dtypes)


@has_schema(
    schema={"new_output": "int", "new_output2": "int"},
    allow_subset=True,
    raise_exc=False,
)
def node_func_bad_schema_no_raise_exc(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_choice_0_disable_raise_exc_w_allow_subset(sample_df_spark_all_dtypes):
    node_func_bad_schema_no_raise_exc(df=sample_df_spark_all_dtypes)


@has_schema(
    schema={"new_output": "int", "new_output2": "int"},
    allow_subset=False,
    raise_exc=False,
)
def node_func_bad_schema_no_raise_exc2(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_choice_0_disable_raise_exc_wo_allow_subset(sample_df_spark_all_dtypes):
    node_func_bad_schema_no_raise_exc2(df=sample_df_spark_all_dtypes)


@has_schema(
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output": "int",
    },
    allow_subset=True,
    raise_exc=True,
    relax=False,
)
def node_func_full_example(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return [df_new, "abc"]


def test_output_full_example(sample_df_spark_all_dtypes):
    node_func_full_example(df=sample_df_spark_all_dtypes)


# ----- pyspark relax -----
@has_schema(
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "timestamp",
        "datetime_col": "timestamp",
        "new_output": "int",
        "array_int": "array<numeric>",
    },
    allow_subset=True,
    raise_exc=True,
    relax=True,
)
def node_func_relax_true(df):
    df_new = df.withColumn("new_output", f.lit(1).cast("long"))
    return [df_new, "abc"]


def test_output_relax_true(sample_df_spark_all_dtypes):
    node_func_relax_true(df=sample_df_spark_all_dtypes)


@has_schema(
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "timestamp",
        "datetime_col": "timestamp",
        "new_output": "int",
    },
    allow_subset=True,
    raise_exc=True,
    relax=False,
)
def node_func_relax_false(df):
    df_new = df.withColumn("new_output", f.lit(1).cast("long"))
    return [df_new, "abc"]


def test_output_relax_false(sample_df_spark_all_dtypes):
    with pytest.raises(ValueError, match=r"Schema differences found"):
        node_func_relax_false(df=sample_df_spark_all_dtypes)


# ----- end -----


@has_schema(
    df="df2",
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "array_int": "array<int>",
        "datetime_col": "timestamp",
    },
    allow_subset=True,
    raise_exc=True,
)
@has_schema(
    df="df1",
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_func_full_example_multiple(df1, df2):
    df1_new = df1.withColumn("new_output", f.lit(1))
    df2_new = df2.withColumn("new_output", f.lit(1))
    return [df1_new, df2_new]


def test_output_full_example_multiple_decorators(sample_df_spark_all_dtypes):
    node_func_full_example_multiple(
        df1=sample_df_spark_all_dtypes, df2=sample_df_spark_all_dtypes
    )


@has_schema(
    output=0,
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output1": "int",
    },
    allow_subset=True,
    raise_exc=True,
)
@has_schema(
    output=1,
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output2": "int",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_func_full_example_multiple_output(df1, df2):
    df1_new = df1.withColumn("new_output1", f.lit(1))
    df2_new = df2.withColumn("new_output2", f.lit(1))
    return [df1_new, df2_new]


def test_output_full_example_multiple_decorators2(sample_df_spark_all_dtypes):
    node_func_full_example_multiple_output(
        df1=sample_df_spark_all_dtypes, df2=sample_df_spark_all_dtypes
    )


@has_schema(
    df="df1",
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    allow_subset=True,
    raise_exc=True,
)
@has_schema(
    output=1,
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output2": "int",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_func_full_example_multiple_input_output(df1, df2):
    df1_new = df1.withColumn("new_output1", f.lit(1))
    df2_new = df2.withColumn("new_output2", f.lit(1))
    return [df1_new, df2_new]


def test_output_full_example_multiple_decorators3(sample_df_spark_all_dtypes):
    node_func_full_example_multiple_input_output(
        df1=sample_df_spark_all_dtypes, df2=sample_df_spark_all_dtypes
    )


# ----- has_schema pandas -----
@pytest.fixture
def sample_df_pd_all_dtypes():
    df = pd.DataFrame(
        {
            "float_col": [1.0],
            "int_col": [1],
            "datetime_col": [pd.Timestamp("20180310")],
            "date_col": [pd.Timestamp("20180310").date()],
            "string_col": ["foo"],
        }
    )

    df["datetime_ms_col"] = df["datetime_col"].values.astype("datetime64[ms]")

    return df


@has_schema(
    output=0,
    df="df",
    schema={"site_id": "int64", "site_name": "string"},
    allow_subset=False,
)
def node_pd_func_return_list(df):
    df_new = df
    return [df_new, "abc"]


def test_pd_has_schema_allow_subset_false(sample_df_pd_all_dtypes):
    with pytest.raises(ValueError, match=r"Please supply only df or output."):
        node_pd_func_return_list(df=sample_df_pd_all_dtypes)


@has_schema(
    schema={"site_id": "int64", "site_name": "string"}, allow_subset=False,
)
def node_pd_func_with_new_col1(df):
    df["new_output"] = 1
    return df


def test_pd_default_to_output_allow_subset_false(sample_df_pd_all_dtypes):
    with pytest.raises(
        AssertionError, match=r"Actual vs expected columns do not match up."
    ):
        node_pd_func_with_new_col1(df=sample_df_pd_all_dtypes)


@has_schema(
    schema={"new_output": "int64"}, allow_subset=True,
)
def node_pd_func_with_new_col2(df):
    df["new_output"] = 1
    return df


def test_pd_default_to_output_allow_subset_true(sample_df_pd_all_dtypes):
    node_pd_func_with_new_col2(df=sample_df_pd_all_dtypes)


@has_schema(
    output=0, schema={"new_output": "int64"}, allow_subset=True,
)
def node_pd_func_return_list1(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_choice_0_dataframe(sample_df_pd_all_dtypes):
    node_pd_func_return_list1(df=sample_df_pd_all_dtypes)


@has_schema(
    output=1, schema={"new_output": "int64"}, allow_subset=True,
)
def node_pd_func_return_list2(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_choice_1_not_dataframe(sample_df_pd_all_dtypes):
    with pytest.raises(ValueError, match=r"should be a spark or pandas dataframe."):
        node_pd_func_return_list2(df=sample_df_pd_all_dtypes)


@has_schema(
    output=0, schema={"new_output": "int64", "new_output2": "int64"}, allow_subset=True,
)
def node_pd_func_bad_schema(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_2(sample_df_pd_all_dtypes):
    with pytest.raises(ValueError, match=r"Schema differences found"):
        node_pd_func_bad_schema(df=sample_df_pd_all_dtypes)


@has_schema(
    schema={"new_output": "int64", "new_output2": "int64"},
    allow_subset=True,
    raise_exc=False,
)
def node_pd_output_choice_0_no_raise_exc(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_choice_0_no_raise_exc_w_allow_subset(sample_df_pd_all_dtypes):
    node_pd_output_choice_0_no_raise_exc(df=sample_df_pd_all_dtypes)


@has_schema(
    schema={"new_output": "int64", "new_output2": "int64"},
    allow_subset=False,
    raise_exc=False,
)
def node_pd_func_bad_schema_no_raise_exc2(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_choice_0_no_raise_exc_wo_allow_subset(sample_df_pd_all_dtypes):
    node_pd_func_bad_schema_no_raise_exc2(df=sample_df_pd_all_dtypes)


@has_schema(
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "new_output": "int64",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_pd_func_full_example(df):
    df["new_output"] = 1
    return [df, "abc"]


def test_pd_output_full_example(sample_df_pd_all_dtypes):
    node_pd_func_full_example(df=sample_df_pd_all_dtypes)


@has_schema(
    df="df1",
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=False,
    raise_exc=True,
)
@has_schema(
    df="df2",
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_pd_func_full_example_multiple(df1, df2):
    df1["new_output"] = 1
    df2["new_output"] = 1
    return [df1, df2]


def test_pd_output_full_example_multiple_decorators(sample_df_pd_all_dtypes):
    node_pd_func_full_example_multiple(
        df1=sample_df_pd_all_dtypes, df2=sample_df_pd_all_dtypes
    )


@has_schema(
    output=0,
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
        "new_output1": "int64",
    },
    allow_subset=False,
    raise_exc=True,
)
@has_schema(
    output=1,
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
        "new_output2": "int64",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_pd_func_full_example_multiple_output(df1, df2):
    df1["new_output1"] = 1
    df2["new_output2"] = 1
    return [df1, df2]


def test_pd_output_full_example_multiple_decorators2(sample_df_pd_all_dtypes):
    # copy otherwise pandas modify the same in memory dataframe
    node_pd_func_full_example_multiple_output(
        df1=sample_df_pd_all_dtypes.copy(), df2=sample_df_pd_all_dtypes.copy()
    )


@has_schema(
    df="df1",
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=True,
    raise_exc=True,
)
@has_schema(
    output=1,
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "datetime_ms_col": "datetime64[ns]",
        "new_output2": "int64",
    },
    allow_subset=False,
    raise_exc=True,
)
def node_pd_func_full_example_multiple_input_output(df1, df2):
    df1["new_output1"] = 1
    df2["new_output2"] = 1
    return [df1, df2]


def test_pd_output_full_example_multiple_decorators3(sample_df_pd_all_dtypes):
    node_pd_func_full_example_multiple_input_output(
        df1=sample_df_pd_all_dtypes.copy(), df2=sample_df_pd_all_dtypes.copy()
    )
