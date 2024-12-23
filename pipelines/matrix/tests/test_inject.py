import inspect
from copy import deepcopy
from types import FunctionType

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import pyspark.sql.functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from matrix.inject import (
    _inject_object,
    _parse_for_objects,
    _unpack_params,
    inject_object,
    make_list_regexable,
    unpack_params,
)


def dummy_func(x):
    """Dummy function for testing purposes."""
    return x


def dummy_func_without_input():
    """Dummy function for testing purposes."""
    return "hello world"


@pytest.fixture
def dummy_pd_df():
    """Dummy pandas dataframe."""
    dummy_pd_df = pd.DataFrame([{"c1": 1}])
    return dummy_pd_df


@pytest.fixture
def param():
    param = {
        "my_imputer": {
            "object": "sklearn.impute.SimpleImputer",
            "strategy": "constant",
            "fill_value": 0,
        },
    }
    return param


@pytest.fixture
def number_as_keys():
    number_as_keys = [
        {
            "my_dict": {
                "default": [3, 4, 5, 6, 7, 8],
                0: [0, 1],
                1: [1, 2, 3],
                2: [1, 2, 3],
                3: [1, 2, 3, 4],
                0.4: [1, 2, 3],
            }
        }
    ]
    return number_as_keys


@pytest.fixture
def nested_params(dummy_pd_df):
    nested_param = {
        "str": "str",
        "int": 0,
        "float": 1.1,
        "z": {
            "object": "tests.test_inject.dummy_func",
            "x": {
                "object": "sklearn.impute.SimpleImputer",
                "strategy": "constant",
                "fill_value": 0,
            },
        },
        "min_max": {
            "object": "sklearn.preprocessing.MinMaxScaler",
            "feature_range": (0, 2),
        },
        "plain_func": {
            "object": "tests.test_inject.dummy_func",
        },
        "inception": {
            "object": "tests.test_inject.dummy_func",
            "x": {"object": "tests.test_inject.dummy_func"},
        },
        "list_of_objs": [
            {"object": "tests.test_inject.dummy_func"},
            {"object": "tests.test_inject.dummy_func"},
            {
                "object": "tests.test_inject.dummy_func",
                "x": {
                    "object": "tests.test_inject.dummy_func",
                },
            },
        ],
        "list_of_objs_nested": [
            {
                "my_func": {
                    "object": "tests.test_inject.dummy_func",
                }
            }
        ],
        "df": dummy_pd_df,
    }
    return nested_param


@pytest.fixture
def another_param():
    return {
        "tuner": {
            "object": "sklearn.model_selection.GridSearchCV",
            "param_grid": {"n_estimators": [5, 10]},
            "estimator": {"object": "sklearn.ensemble.RandomForestRegressor"},
            "cv": {
                "object": "sklearn.model_selection.ShuffleSplit",
                "random_state": 1,
            },
        }
    }


@pytest.fixture
def flat_params():
    flat_params = [
        {"object": "tests.test_inject.dummy_func"},
        {"object": "tests.test_inject.dummy_func"},
    ]
    return flat_params


@inject_object()
def my_func_pd_imputer(*args, **kwargs):
    def dummy_func(df, x, y, imputer):
        df["new"] = df["c1"] + x + y
        return df, imputer

    return dummy_func(*args, **kwargs)


def test_parse_for_objects_class(param):
    result = _parse_for_objects(param)
    assert isinstance(result["my_imputer"], SimpleImputer)


def test_nested_params_in_parse_for_objects(nested_params):
    fake_df = pd.DataFrame([{"c1": 1}])

    result = _parse_for_objects({"fake_df": fake_df, **nested_params})
    assert isinstance(result["fake_df"], pd.DataFrame)

    assert result["str"] == "str"
    assert result["int"] == 0
    assert result["float"] == 1.1
    assert isinstance(result["z"], SimpleImputer)
    assert isinstance(result["min_max"], MinMaxScaler)
    assert isinstance(result["min_max"].feature_range, tuple)
    assert result["plain_func"].__name__ == dummy_func.__name__
    assert result["inception"].__name__ == dummy_func.__name__
    assert (
        result["list_of_objs"][0].__name__
        == result["list_of_objs"][1].__name__
        == result["list_of_objs"][2].__name__
        == dummy_func.__name__
    )
    assert result["list_of_objs_nested"][0]["my_func"].__name__ == dummy_func.__name__


def test_flat_params_in_parse_for_objects(flat_params):
    result = _parse_for_objects(flat_params)
    assert result[0].__name__ == dummy_func.__name__
    assert result[1].__name__ == dummy_func.__name__


def test_numbers_as_keys(number_as_keys):
    new_args, new_kwargs = _inject_object(*number_as_keys)
    assert new_args[0] == number_as_keys[0]


def test_invalid_keyword_in_parse_for_objects():
    invalid_param = {"object_invalid": "tests.test_inject.dummy_func"}
    result = _parse_for_objects(invalid_param)
    assert isinstance(result["object_invalid"], str)  # it should not load object


def test_instantiate_function_without_input():
    instantiate_param = {
        "object": "tests.test_inject.dummy_func_without_input",
        "instantiate": True,
    }
    no_instantiate_param = {
        "object": "tests.test_inject.dummy_func_without_input",
        "instantiate": False,
    }
    default_param = {
        "object": "tests.test_inject.dummy_func_without_input",
    }

    result1 = _parse_for_objects(instantiate_param)
    result2 = _parse_for_objects(no_instantiate_param)
    result3 = _parse_for_objects(default_param)

    assert result1 == "hello world"
    assert isinstance(result2, FunctionType)
    assert isinstance(result3, FunctionType)


def test_instantiate_function_with_input():
    instantiate_param = {
        "object": "tests.test_inject.dummy_func",
        "x": 1,
        "instantiate": True,
    }
    no_instantiate_param = {
        "object": "tests.test_inject.dummy_func",
        "instantiate": False,
    }
    default_param = {
        "object": "tests.test_inject.dummy_func",
        "x": 1,
    }
    default_param_with_tuple = {
        "object": "tests.test_inject.dummy_func",
        "x": (1, 2, 3),
    }

    result1 = _parse_for_objects(instantiate_param)
    result2 = _parse_for_objects(no_instantiate_param)
    result3 = _parse_for_objects(default_param)
    result4 = _parse_for_objects(default_param_with_tuple)

    assert result1 == 1
    assert isinstance(result2, FunctionType)
    assert result3 == 1
    assert result4 == (1, 2, 3)


def test_instantiate_class():
    instantiate_param = {
        "object": "sklearn.impute.SimpleImputer",
        "instantiate": True,
    }
    no_instantiate_param = {
        "object": "sklearn.impute.SimpleImputer",
        "instantiate": False,
    }
    default_param = {
        "object": "sklearn.impute.SimpleImputer",
    }

    result1 = _parse_for_objects(instantiate_param)
    result2 = _parse_for_objects(no_instantiate_param)
    result3 = _parse_for_objects(default_param)

    assert isinstance(result1, SimpleImputer)
    assert inspect.isclass(result2)
    assert isinstance(result3, SimpleImputer)


def test_inject_class(param):
    new_args, new_kwargs = _inject_object(**param)
    assert not new_args
    assert isinstance(new_kwargs["my_imputer"], SimpleImputer)


def test_nested_params(nested_params):
    fake_df = pd.DataFrame([{"c1": 1}])

    new_args, new_kwargs = _inject_object(**{"fake_df": fake_df, **nested_params})
    assert not new_args
    assert isinstance(new_kwargs["fake_df"], pd.DataFrame)

    assert new_kwargs["str"] == "str"
    assert new_kwargs["int"] == 0
    assert new_kwargs["float"] == 1.1
    assert isinstance(new_kwargs["z"], SimpleImputer)
    assert new_kwargs["plain_func"].__name__ == dummy_func.__name__
    assert new_kwargs["inception"].__name__ == dummy_func.__name__
    assert (
        new_kwargs["list_of_objs"][0].__name__
        == new_kwargs["list_of_objs"][1].__name__
        == new_kwargs["list_of_objs"][2].__name__
        == dummy_func.__name__
    )
    assert new_kwargs["list_of_objs_nested"][0]["my_func"].__name__ == dummy_func.__name__
    assert isinstance(new_kwargs["df"], pd.DataFrame)


def test_exclude_kwargs_as_dict_key(nested_params):
    _, new_kwargs = _inject_object(**nested_params, exclude_kwargs=["z"])
    assert "z" in new_kwargs.keys()
    assert "object" in new_kwargs["z"].keys()


def test_exclude_kwargs_as_params(another_param):
    _, new_kwargs = _inject_object(**another_param, exclude_kwargs=["cv", "estimator"])

    # test estimator key still is in inject syntax
    tuner_object = new_kwargs["tuner"]
    assert hasattr(tuner_object, "estimator")
    assert "object" in tuner_object.estimator.keys()

    # test cv key still is in inject syntax
    assert hasattr(tuner_object, "cv")
    assert "object" in tuner_object.cv.keys()


def test_config_is_not_mutated(nested_params):
    nested_params_copy = deepcopy(nested_params)
    _inject_object(**nested_params)
    df_copy = nested_params_copy.pop("df")
    df = nested_params.pop("df")
    assert nested_params_copy == nested_params
    pd.testing.assert_frame_equal(df, df_copy)


def test_flat_params_in_inject_object(flat_params):
    new_args, new_kwargs = _inject_object(*flat_params)
    assert not new_kwargs
    assert new_args[0].__name__ == dummy_func.__name__
    assert new_args[1].__name__ == dummy_func.__name__


def test_invalid_keyword_in_inject_object():
    invalid_param = {"object_invalid": "tests.test_inject.dummy_func"}
    new_args, new_kwargs = _inject_object(**invalid_param)
    assert not new_args
    assert isinstance(new_kwargs["object_invalid"], str)  # it should not load objects


def test_additional_params_in_inject_object(another_param):
    new_args, new_kwargs = _inject_object(another_param)
    assert new_args[0]
    assert isinstance(new_args[0]["tuner"], GridSearchCV)


def test_inject(dummy_pd_df):
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


@pytest.fixture
def pandas_df():
    """Sample pandas dataframe with all dtypes."""
    data = [
        {
            "float_col": 1.0,
            "int_col": 1,
            "string_col": "foo",
        },
        {
            "float_col": 1.0,
            "int_col": 2,
            "string_col": "blabla",
        },
        {
            "float_col": 1.0,
            "int_col": 3,
            "string_col": None,
        },
    ]
    df = pd.DataFrame(data)

    return df


def test_make_list_regexable_with_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    # Pass arguments positionally instead of as keywords
    result_df = accept_regexable_list(
        pandas_df,
        [".*col"],
    )

    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_with_explicit_names_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    result_df = accept_regexable_list(
        pandas_df,  # df
        ["int_col", "float_col", "string_col"],  # params_keep_cols
    )
    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_with_combination_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    result_df = accept_regexable_list(
        pandas_df,
        ["int_.*", "float_col", "string_col"],
    )
    assert len(result_df.columns) == len(pandas_df.columns)


@pytest.fixture
def spark_df(spark):
    """Sample spark dataframe with all dtypes."""
    schema = StructType(
        [
            StructField("int_col", IntegerType(), True),
            StructField("float_col", FloatType(), True),
            StructField("string_col", StringType(), True),
        ]
    )

    data = [
        (
            1,
            2.0,
            "awesome string",
        ),
        (
            2,
            2.0,
            None,
        ),
        (
            3,
            2.0,
            "hello world",
        ),
    ]

    return spark.createDataFrame(data, schema)


def test_make_list_regexable_spark(spark_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df.select(*params_keep_cols)
        return result_df

    result_df = accept_regexable_list(
        spark_df,
        [".*col"],
    )
    assert len(result_df.columns) == len(spark_df.columns)


def test_make_list_regexable_with_explicit_names_spark(spark_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df.select(*params_keep_cols)
        return result_df

    result_df = accept_regexable_list(
        spark_df,
        ["int_col", "float_col", "string_col"],
    )
    assert len(result_df.columns) == len(spark_df.columns)


def test_make_list_regexable_with_combination_spark(spark_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df.select(*params_keep_cols)
        return result_df

    result_df = accept_regexable_list(
        spark_df,
        ["int_.*", "float_col", "string_col"],
    )
    assert len(result_df.columns) == len(spark_df.columns)


def test_raise_exc_default_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    with pytest.raises(ValueError, match="No columns were selected using the provided regex patterns*"):
        accept_regexable_list(pandas_df, ["notfloat.*"])


def test_raise_exc_enabled_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
        raise_exc=True,
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    with pytest.raises(
        ValueError,
        match="The following regex did not return a result: ftr1_.*.",
    ):
        accept_regexable_list(pandas_df, ["ftr1_.*"])


def test_make_list_regexable_accept_args_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    result_df = accept_regexable_list(pandas_df, [".*col"])
    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_not_present_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df):
        result_df = df
        return result_df

    result_df = accept_regexable_list(pandas_df)
    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_empty_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df
        return result_df

    with pytest.raises(ValueError, match="No columns were selected using the provided regex patterns*"):
        accept_regexable_list(pandas_df, [])


def test_make_list_regexable_source_df_not_present_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(params_keep_cols):
        params_keep_cols = params_keep_cols
        return params_keep_cols

    with pytest.raises(
        ValueError,
        match="Please provide source dataframe",
    ):
        accept_regexable_list(["col.*"])


def test_make_list_regexable_with_wrong_input_type_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_column="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    with pytest.raises(
        TypeError,
        match="'int' object is not iterable",
    ):
        accept_regexable_list(pandas_df, 7)


@unpack_params()
def my_func(*args, **kwargs):
    def test_func(x):
        return x["c1"]

    return test_func(*args, **kwargs)


def test_unpack_params_with_kwargs():
    result_arg, result_kwarg = _unpack_params(unpack={"c1": 1})

    assert result_arg == []
    assert result_kwarg == {"c1": 1}


def test_unpack_params_with_args():
    @unpack_params()
    def dummy_func(x, y, z, params):
        params["test"] == "test"
        return x + y - z

    x = 1
    param = {"params": {"test": "test"}, "unpack": {"y": 1, "z": 2}}
    result = dummy_func(x, param)
    assert result == 0


@unpack_params()
def dummy_func2(x, y, z, params1, params2):
    params1["test"] == 1
    params2["test"] == 2
    return x + y - z


def test_unpack_params_with_multiple_args():
    x = 1
    param1 = {"params1": {"test": 1}, "unpack": {"y": 2}}
    param2 = {"params2": {"test": 2}, "unpack": {"z": 3}}
    result = dummy_func2(x, param1, param2)
    assert result == 0


def test_unpack_params_with_multiple_args_and_kwargs():
    param1 = {"params1": {"test": 1}, "unpack": {"y": 2}}
    param2 = {"params2": {"test": 2}, "unpack": {"z": 3}}
    result = dummy_func2(param1, param2, unpack={"x": 1})
    assert result == 0


def test_unpack_params_disable_via_args():
    result_arg, result_kwarg = _unpack_params({"c1": 1})

    assert result_arg == [
        {"c1": 1},
    ]
    assert result_kwarg == {}


def test_unpack_params_disable_via_args2():
    result_arg, result_kwarg = _unpack_params(**{"c1": 1})

    assert result_arg == []
    assert result_kwarg == {"c1": 1}


def test_unpack_params_disable_via_kwargs():
    result_arg, result_kwarg = _unpack_params(x={"c1": 1})

    assert result_arg == []
    assert result_kwarg == {"x": {"c1": 1}}


def test_unpack_params_with_pd_df(dummy_pd_df):
    result_arg, result_kwarg = _unpack_params(unpack={"df": dummy_pd_df, "x": 1})

    def dummy_func(df, x):
        df["new"] = df["c1"] + x
        return df

    result = dummy_func(*result_arg, **result_kwarg)
    assert result["new"].tolist() == [2]


@pytest.fixture
def dummy_spark_df(spark):
    """Dummy spark dataframe."""
    dummy_pd_df = pd.DataFrame([{"c1": 1}])
    dummy_spark_df = spark.createDataFrame(dummy_pd_df)
    return dummy_spark_df


def test_unpack_params_with_spark_df(dummy_spark_df):
    result_arg, result_kwarg = _unpack_params(unpack={"df": dummy_spark_df, "x": 1})

    def dummy_func(df, x):
        df = df.withColumn("new", F.lit(x) + 1)
        return df

    result = dummy_func(*result_arg, **result_kwarg)
    assert [x.asDict() for x in result.select("new").collect()] == [{"new": 2}]


def test_unpack_params_true_decorator():
    result = my_func(unpack={"x": {"c1": 1}})
    assert result == 1


def test_unpack_params_false_decorator():
    with pytest.raises(KeyError, match="c1"):
        my_func({"x": {"c1": 1}})


@make_list_regexable(source_df="df", make_regexable_column="columns")
def dummy_function(df, columns):
    """
    A simple function to test the make_list_regexable decorator when passing arguments as kwargs.
    In theory, if make_list_regexable handled kwargs the same as positional args, this function
    should accept df and columns as kwargs without issue.
    """
    return df, columns


@pytest.mark.parametrize("arg_style", ["all_positional", "all_kwargs", "mixed"])
def test_make_list_regexable_with_varied_args(arg_style, dummy_pd_df):
    df = pd.DataFrame({"some_col": [1], "other_col": [2]})
    columns = ["^some_.*"]
    expected_result = (df, ["some_col"])

    if arg_style == "all_positional":
        result = dummy_function(df, columns)
    elif arg_style == "all_kwargs":
        result = dummy_function(df=df, columns=columns)
    elif arg_style == "mixed":
        result = dummy_function(df, columns=columns)

    assert isinstance(result, tuple), "Function should return a tuple of (df, columns)"
    result_df, result_cols = result
    pd.testing.assert_frame_equal(result_df, expected_result[0])
    assert result_cols == expected_result[1]
