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
"""Tests for primary_key."""

import pandas as pd
import pytest

from refit.v1.core.primary_key import INPUT_PRIMARY_KEY, OUTPUT_PRIMARY_KEY, primary_key


@pytest.fixture
def sample_pd():
    """Sample dictionary for pandas table."""
    return {
        "group": ["Group A", "Group A", "Group B", "Group C"],
        "sub_group": ["US", "UK", "EU", "US"],
        "revenue": [2.2, 1.2, 3.8, 1.5],
    }


@pytest.fixture
def sample_spark():
    """Sample dictionary for pyspark table."""
    return {
        "group": ["Group A", "Group A", "Group A", "Group B"],
        "sub_group": ["US", "UK", "US", "US"],
        "sub_sub_group": ["id1", "id2", "id3", "id4"],
        "capacity": [20, 49, 38, 55],
    }


@pytest.fixture
def sample_df_pd(sample_pd):
    df = pd.DataFrame(sample_pd)
    return df


@pytest.fixture
def sample_df_spark(sample_spark, spark):
    df_pd = pd.DataFrame(sample_spark)
    df = spark.createDataFrame(df_pd)
    return df


@pytest.fixture
def sample_dict():
    return {"sample_key": "sample_value"}


@primary_key()
def list_no_input_no_output_func():
    print("This is just a dummy function")


@primary_key()
def list_single_input_no_output_func(df1):
    print("This is just a dummy function")


@primary_key()
def list_multiple_input_no_output_func(df1, df2, dict1):
    print("This is just a dummy function")


@primary_key()
def list_no_input_single_output_func():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7], "value": [1, 1, 1, 0, 0, 1, 0]})
    return df


@primary_key()
def list_no_input_multiple_output_func():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7], "value": [1, 1, 1, 0, 0, 1, 0]})
    df2 = pd.DataFrame(
        {
            "id1": [1, 1, 2, 2, 3, 4, 4],
            "id2": [1, 2, 1, 2, 2, 2, 3],
            "value": [1, 1, 1, 0, 0, 1, 0],
        }
    )
    dict1 = {"sample_key": "sample_value"}
    return df1, df2, dict1


@primary_key()
def list_single_input_single_return_func(df1):
    return df1


@primary_key()
def list_multiple_input_multiple_return_func(df1, df2, dict1):
    return df2, df1, dict1


class TestListOutputs:
    def test_pk_list_no_input_no_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and output function with input_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            list_no_input_no_output_func(input_primary_key={"columns": []})

    def test_pk_list_no_input_no_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and output function with output_primary_key."""
        with pytest.raises(
            ValueError, match="`result_df` should be of type pandas or spark dataframe."
        ):
            list_no_input_no_output_func(output_primary_key={"columns": []})

    def test_pk_list_no_input_no_output_func_with_both_input_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and output function with both input_primary_key and output_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            list_no_input_no_output_func(
                input_primary_key={"columns": []}, output_primary_key={"columns": []}
            )

    def test_pk_list_no_input_no_output_func_with_no_input_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and output function with no input_primary_key or output_primary_key."""
        list_no_input_no_output_func()

    def test_pk_list_single_input_no_output_func_with_single_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input (Pandas) and no output function with input_primary_key."""
        list_single_input_no_output_func(
            sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
        )

    def test_pk_list_single_input_no_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input (Pandas) and no output function with output_primary_key."""
        with pytest.raises(
            ValueError, match="`result_df` should be of type pandas or spark dataframe."
        ):
            list_single_input_no_output_func(
                sample_df_pd, output_primary_key={"columns": ["group", "sub_group"]}
            )

    def test_pk_list_multiple_input_no_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input (Pandas and Spark) and no output function with input_primary_key."""
        list_multiple_input_no_output_func(
            sample_df_pd,
            sample_df_spark,
            sample_dict,
            input_primary_key=[
                {"columns": ["group", "sub_group"], "index": 0},
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": 1},
            ],
        )
        list_multiple_input_no_output_func(
            sample_df_pd,
            sample_df_spark,
            sample_dict,
            input_primary_key=[{"columns": ["group", "sub_group"], "index": 0}],
        )

    def test_pk_list_multiple_input_no_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input (Pandas and Spark) and no output function with output_primary_key."""
        with pytest.raises(
            ValueError, match="`result_df` should be of type pandas or spark dataframe."
        ):
            list_multiple_input_no_output_func(
                sample_df_pd,
                sample_df_spark,
                sample_dict,
                output_primary_key=[{"columns": ["group", "sub_group"], "index": 0},],
            )

    def test_pk_list_no_input_single_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Single output function with input_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            returned_df = list_no_input_single_output_func(
                input_primary_key={"columns": ["id"]}
            )

    def test_pk_list_no_input_single_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Single output function with output_primary_key."""
        returned_df = list_no_input_single_output_func(
            output_primary_key={"columns": ["id"]}
        )
        assert isinstance(returned_df, pd.DataFrame)

    def test_pk_list_no_input_multiple_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Multiple (Pandas) output function with input_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            df1, df2, dict1 = list_no_input_multiple_output_func(
                input_primary_key={"columns": []}
            )

    def test_pk_list_no_input_multiple_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Multiple (Pandas) output function with output_primary_key."""
        df1, df2, dict1 = list_no_input_multiple_output_func(
            output_primary_key=[
                {"columns": ["id"], "index": 0},
                {"columns": ["id1", "id2"], "index": 1},
            ]
        )
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert dict1 == sample_dict

    def test_pk_list_single_input_single_pd_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input and Single (Pandas) output function with input_primary_key."""
        returned_df = list_single_input_single_return_func(
            sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
        )
        assert returned_df.equals(sample_df_pd)

    def test_pk_list_single_input_single_spark_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input and Single (Spark) output function with output_primary_key."""
        returned_df = list_single_input_single_return_func(
            sample_df_spark,
            output_primary_key={"columns": ["group", "sub_group", "sub_sub_group"]},
        )
        assert (returned_df.schema == sample_df_spark.schema) and (
            returned_df.collect() == sample_df_spark.collect()
        )

    def test_pk_list_single_input_single_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Scenario: Single input and Single (Pandas) output function with output_primary_key."""
        returned_df = list_single_input_single_return_func(
            sample_df_pd,
            input_primary_key={"columns": ["group", "sub_group"]},
            output_primary_key={"columns": ["group", "sub_group"]},
        )
        assert returned_df.equals(sample_df_pd)

    def test_pk_list_multiple_input_multiple_output_func_with_input_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input and Multiple output function with both input_primary_key and output_primary_key."""
        df1, df2, dict1 = list_multiple_input_multiple_return_func(
            sample_df_pd,
            sample_df_spark,
            sample_dict,
            input_primary_key=[
                {"columns": ["group", "sub_group"], "index": 0},
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": 1},
            ],
            output_primary_key=[
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": 0},
                {"columns": ["group", "sub_group"], "index": 1},
            ],
        )
        assert (df1.schema == sample_df_spark.schema) and (
            df1.collect() == sample_df_spark.collect()
        )
        assert df2.equals(sample_df_pd)
        assert dict1 == sample_dict


# Same as above just with kwargs and dictionary return type --------------------------
@primary_key()
def dict_single_input_no_output_func(df1):
    print("This is just a dummy function")


@primary_key()
def dict_multiple_input_no_output_func(df1, df2, dict1):
    print("This is just a dummy function")


@primary_key()
def dict_no_input_single_output_func():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7], "value": [1, 1, 1, 0, 0, 1, 0]})
    return {"df1": df1}


@primary_key()
def dict_no_input_multiple_output_func():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7], "value": [1, 1, 1, 0, 0, 1, 0]})
    df2 = pd.DataFrame(
        {
            "id1": [1, 1, 2, 2, 3, 4, 4],
            "id2": [1, 2, 1, 2, 2, 2, 3],
            "value": [1, 1, 1, 0, 0, 1, 0],
        }
    )
    dict1 = {"sample_key": "sample_value"}
    return {"df1": df1, "df2": df2, "dict1": dict1}


@primary_key()
def dict_single_input_single_return_func(df1):
    return {"df1": df1}


@primary_key()
def dict_multiple_input_multiple_return_func(df1, df2, dict1):
    return {"df2": df2, "df1": df1, "dict1": dict1}


class TestDictOutputs:
    def test_pk_dict_single_input_no_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input (Pandas) and no output function with input_primary_key."""
        dict_single_input_no_output_func(
            df1=sample_df_pd,
            input_primary_key={"columns": ["group", "sub_group"], "index": "df1"},
        )

    def test_pk_dict_single_input_no_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input (Pandas) and no output function with output_primary_key."""
        with pytest.raises(
            ValueError, match="`result_df` should be of type pandas or spark dataframe."
        ):
            dict_single_input_no_output_func(
                df1=sample_df_pd,
                output_primary_key={"columns": ["group", "sub_group"]},
            )

    def test_pk_dict_multiple_input_no_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input (Pandas and Spark) and no output function with input_primary_key."""
        dict_multiple_input_no_output_func(
            df1=sample_df_pd,
            df2=sample_df_spark,
            dict1=sample_dict,
            input_primary_key=[
                {"columns": ["group", "sub_group"], "index": "df1"},
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": "df2"},
            ],
        )
        dict_multiple_input_no_output_func(
            df1=sample_df_pd,
            df2=sample_df_spark,
            dict1=sample_dict,
            input_primary_key=[{"columns": ["group", "sub_group"], "index": "df1"}],
        )

    def test_pk_dict_multiple_input_no_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input (Pandas and Spark) and no output function with output_primary_key."""
        with pytest.raises(
            ValueError, match="`result_df` should be of type pandas or spark dataframe."
        ):
            dict_multiple_input_no_output_func(
                df1=sample_df_pd,
                df2=sample_df_spark,
                dict1=sample_dict,
                output_primary_key=[
                    {"columns": ["group", "sub_group"], "index": "df1"},
                ],
            )

    def test_pk_dict_no_input_single_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Single output function with input_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            returned_dict = dict_no_input_single_output_func(
                input_primary_key={"columns": ["id"]}
            )

    def test_pk_dict_no_input_single_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Single output function with output_primary_key."""
        returned_dict = dict_no_input_single_output_func(
            output_primary_key={"columns": ["id"], "index": "df1"}
        )
        assert isinstance(returned_dict["df1"], pd.DataFrame)

    def test_pk_dict_no_input_multiple_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Multiple (Pandas) output function with input_primary_key."""
        with pytest.raises(
            ValueError,
            match=f"`{INPUT_PRIMARY_KEY}` was used but no args and kwargs passed.",
        ):
            returned_dict = dict_no_input_multiple_output_func(
                input_primary_key={"columns": []}
            )

    def test_pk_dict_no_input_multiple_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """No input and Multiple (Pandas) output function with output_primary_key."""
        returned_dict = dict_no_input_multiple_output_func(
            output_primary_key=[
                {"columns": ["id"], "index": "df1"},
                {"columns": ["id1", "id2"], "index": "df2"},
            ]
        )
        assert isinstance(returned_dict["df1"], pd.DataFrame)
        assert isinstance(returned_dict["df2"], pd.DataFrame)
        assert returned_dict["dict1"] == sample_dict

    def test_pk_dict_single_input_single_output_func_with_input_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input and Single (Pandas) output function with input_primary_key."""
        returned_dict = dict_single_input_single_return_func(
            df1=sample_df_pd,
            input_primary_key={"columns": ["group", "sub_group"], "index": "df1"},
        )
        assert returned_dict["df1"].equals(sample_df_pd)

    def test_pk_dict_single_input_single_spark_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input and Single (Spark) output function with output_primary_key."""
        returned_dict = dict_single_input_single_return_func(
            df1=sample_df_spark,
            output_primary_key={
                "columns": ["group", "sub_group", "sub_sub_group"],
                "index": "df1",
            },
        )
        assert (returned_dict["df1"].schema == sample_df_spark.schema) and (
            returned_dict["df1"].collect() == sample_df_spark.collect()
        )

    def test_pk_dict_single_input_single_pd_output_func_with_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Single input and Single (Pandas) output function with output_primary_key."""
        returned_dict = dict_single_input_single_return_func(
            df1=sample_df_pd,
            input_primary_key={"columns": ["group", "sub_group"], "index": "df1"},
            output_primary_key={"columns": ["group", "sub_group"], "index": "df1"},
        )
        assert returned_dict["df1"].equals(sample_df_pd)

    def test_pk_dict_multiple_input_multiple_output_func_with_input_output_primary_key(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Multiple input and Multiple output function with both input_primary_key and output_primary_key."""
        returned_dict = dict_multiple_input_multiple_return_func(
            df1=sample_df_pd,
            df2=sample_df_spark,
            dict1=sample_dict,
            input_primary_key=[
                {"columns": ["group", "sub_group"], "index": "df1"},
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": "df2"},
            ],
            output_primary_key=[
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": "df2"},
                {"columns": ["group", "sub_group"], "index": "df1"},
            ],
        )
        assert returned_dict["df1"].equals(sample_df_pd)
        assert (returned_dict["df2"].schema == sample_df_spark.schema) and (
            returned_dict["df2"].collect() == sample_df_spark.collect()
        )
        assert returned_dict["dict1"] == sample_dict

    def test_primary_key_dict_mix_args_kwargs(
        self, sample_df_pd, sample_df_spark, sample_dict
    ):
        """Test all scenarios for dict returns and mixed args, kwargs inputs."""
        with pytest.raises(
            ValueError,
            match=f"Please use either args or kwargs with `{INPUT_PRIMARY_KEY}`.",
        ):
            returned_dict = dict_multiple_input_multiple_return_func(
                sample_df_pd,
                df2=sample_df_spark,
                dict1=sample_dict,
                input_primary_key=[
                    {"columns": ["group", "sub_group"], "index": 0},
                    {
                        "columns": ["group", "sub_group", "sub_sub_group"],
                        "index": "df2",
                    },
                ],
                output_primary_key=[
                    {
                        "columns": ["group", "sub_group", "sub_sub_group"],
                        "index": "df2",
                    },
                    {"columns": ["group", "sub_group"], "index": "df1"},
                ],
            )

        returned_dict = dict_multiple_input_multiple_return_func(
            sample_df_pd,
            df2=sample_df_spark,
            dict1=sample_dict,
            output_primary_key=[
                {"columns": ["group", "sub_group", "sub_sub_group"], "index": "df2"},
                {"columns": ["group", "sub_group"], "index": "df1"},
            ],
        )
        assert returned_dict["df1"].equals(sample_df_pd)
        assert (returned_dict["df2"].schema == sample_df_spark.schema) and (
            returned_dict["df2"].collect() == sample_df_spark.collect()
        )
        assert returned_dict["dict1"] == sample_dict


def test_single_input_with_input_primary_key_no_index(
    sample_df_pd, sample_df_spark, sample_dict
):
    """Test single input primary key with no index for the config."""

    dict_single_input_no_output_func(
        df1=sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
    )

    dict_single_input_no_output_func(
        sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
    )

    list_single_input_no_output_func(
        df1=sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
    )

    list_single_input_no_output_func(
        sample_df_pd, input_primary_key={"columns": ["group", "sub_group"]}
    )
