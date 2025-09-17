from copy import deepcopy
from types import FunctionType

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest
from matrix_inject.inject import (
    _inject_object,
    _parse_for_objects,
    _unpack_params,
    inject_object,
    make_list_regexable,
    unpack_params,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


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
            "_object": "sklearn.impute.SimpleImputer",
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
            "_object": "tests.test_inject.dummy_func",
            "x": {
                "_object": "sklearn.impute.SimpleImputer",
                "strategy": "constant",
                "fill_value": 0,
            },
        },
        "min_max": {
            "_object": "sklearn.preprocessing.MinMaxScaler",
            "feature_range": (0, 2),
        },
        "plain_func": {
            "_object": "tests.test_inject.dummy_func",
        },
        "inception": {
            "_object": "tests.test_inject.dummy_func",
            "x": {"_object": "tests.test_inject.dummy_func"},
        },
        "list_of_objs": [
            {"_object": "tests.test_inject.dummy_func"},
            {"_object": "tests.test_inject.dummy_func"},
            {
                "_object": "tests.test_inject.dummy_func",
                "x": {
                    "_object": "tests.test_inject.dummy_func",
                },
            },
        ],
        "list_of_objs_nested": [
            {
                "my_func": {
                    "_object": "tests.test_inject.dummy_func",
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
            "_object": "sklearn.model_selection.GridSearchCV",
            "param_grid": {"n_estimators": [5, 10]},
            "estimator": {"_object": "sklearn.ensemble.RandomForestRegressor"},
            "cv": {
                "_object": "sklearn.model_selection.ShuffleSplit",
                "random_state": 1,
            },
        }
    }


@pytest.fixture
def flat_params():
    flat_params = [
        {"_object": "tests.test_inject.dummy_func"},
        {"_object": "tests.test_inject.dummy_func"},
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


@pytest.mark.parametrize(
    "test_type, param, expected_type, expected_value",
    [
        (
            "function_without_input",
            {
                "_object": "tests.test_inject.dummy_func_without_input",
                "instantiate": True,
            },
            str,
            "hello world",
        ),
        (
            "function_without_input",
            {
                "_object": "tests.test_inject.dummy_func_without_input",
                "instantiate": False,
            },
            FunctionType,
            None,
        ),
        (
            "function_with_input",
            {
                "_object": "tests.test_inject.dummy_func",
                "x": 1,
                "instantiate": True,
            },
            int,
            1,
        ),
        (
            "function_with_input",
            {
                "_object": "tests.test_inject.dummy_func",
                "instantiate": False,
            },
            FunctionType,
            None,
        ),
        (
            "class",
            {
                "_object": "sklearn.impute.SimpleImputer",
                "instantiate": True,
            },
            SimpleImputer,
            None,
        ),
        (
            "class",
            {
                "_object": "sklearn.impute.SimpleImputer",
                "instantiate": False,
            },
            type,
            None,
        ),
    ],
    ids=[
        "func_no_input_true",
        "func_no_input_false",
        "func_input_true",
        "func_input_false",
        "class_true",
        "class_false",
    ],
)
def test_instantiate_behavior(test_type, param, expected_type, expected_value):
    result = _parse_for_objects(param)
    assert isinstance(result, expected_type)
    if expected_value is not None:
        assert result == expected_value


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
    assert "_object" in new_kwargs["z"].keys()


def test_exclude_kwargs_as_params(another_param):
    _, new_kwargs = _inject_object(**another_param, exclude_kwargs=["cv", "estimator"])

    # test estimator key still is in inject syntax
    tuner_object = new_kwargs["tuner"]
    assert hasattr(tuner_object, "estimator")
    assert "_object" in tuner_object.estimator.keys()

    # test cv key still is in inject syntax
    assert hasattr(tuner_object, "cv")
    assert "_object" in tuner_object.cv.keys()


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
            "imputer": {"_object": "sklearn.impute.SimpleImputer"},
        }
    )
    assert isinstance(imputer, SimpleImputer)

    assert df["new"].tolist() == [2]


@pytest.mark.parametrize(
    "df_fixture, select_func",
    [
        ("pandas_df", lambda df, cols: df[cols]),
        ("spark_df", lambda df, cols: df.select(*cols)),
    ],
    ids=["pandas", "spark"],
)
def test_make_list_regexable(request, df_fixture, select_func):
    df = request.getfixturevalue(df_fixture)

    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = select_func(df, params_keep_cols)
        return result_df

    # Test with regex pattern
    result_df = accept_regexable_list(df, [".*col"])
    assert len(result_df.columns) == len(df.columns)

    # Test with explicit names
    result_df = accept_regexable_list(
        df,
        ["int_col", "float_col", "string_col"],
    )
    assert len(result_df.columns) == len(df.columns)

    # Test with combination
    result_df = accept_regexable_list(
        df,
        ["int_.*", "float_col", "string_col"],
    )
    assert len(result_df.columns) == len(df.columns)


@pytest.fixture()
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


def test_make_list_regexable_with_explicit_names_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
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
        make_regexable_kwarg="params_keep_cols",
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
    schema = T.StructType(
        [
            T.StructField("int_col", T.IntegerType(), True),
            T.StructField("float_col", T.FloatType(), True),
            T.StructField("string_col", T.StringType(), True),
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


def test_make_list_regexable_with_explicit_names_spark(spark_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
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
        make_regexable_kwarg="params_keep_cols",
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
        make_regexable_kwarg="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    with pytest.raises(ValueError, match="No columns were selected using the provided regex patterns*"):
        accept_regexable_list(pandas_df, ["notfloat.*"])


def test_raise_exc_enabled_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
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
        make_regexable_kwarg="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df[params_keep_cols]
        return result_df

    result_df = accept_regexable_list(pandas_df, [".*col"])
    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_not_present_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
    )
    def accept_regexable_list(df):
        result_df = df
        return result_df

    result_df = accept_regexable_list(pandas_df)
    assert len(result_df.columns) == len(pandas_df.columns)


def test_make_list_regexable_empty_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
    )
    def accept_regexable_list(df, params_keep_cols):
        result_df = df
        return result_df

    with pytest.raises(ValueError, match="No columns were selected using the provided regex patterns*"):
        accept_regexable_list(pandas_df, [])


def test_make_list_regexable_source_df_not_present_pandas(pandas_df):
    @make_list_regexable(
        source_df="df",
        make_regexable_kwarg="params_keep_cols",
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
        make_regexable_kwarg="params_keep_cols",
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


@pytest.mark.parametrize(
    "test_case,args,kwargs,expected",
    [
        (
            "args_only",
            [{"params1": {"test": 1}, "unpack": {"y": 2}}, {"params2": {"test": 2}, "unpack": {"z": 3}}],
            {"x": 1},
            0,
        ),
        ("kwargs_only", [], {"params1": {"test": 1}, "unpack": {"y": 2}, "params2": {"test": 2}, "z": 3, "x": 1}, 0),
    ],
    ids=["args", "kwargs"],
)
def test_unpack_params_multiple(test_case, args, kwargs, expected):
    result = dummy_func2(*args, **kwargs)
    assert result == expected


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


@pytest.fixture
def dummy_spark_df(spark):
    """Dummy spark dataframe."""
    dummy_pd_df = pd.DataFrame([{"c1": 1}])
    dummy_spark_df = spark.createDataFrame(dummy_pd_df)
    return dummy_spark_df


@pytest.mark.parametrize(
    "df_fixture,transform_func,expected",
    [
        (
            "dummy_pd_df",
            lambda df, x: df.assign(new=df["c1"] + x),
            lambda result: result["new"].tolist() == [2],
        ),
        (
            "dummy_spark_df",
            lambda df, x: df.withColumn("new", F.lit(x) + 1),
            lambda result: [x.asDict() for x in result.select("new").collect()] == [{"new": 2}],
        ),
    ],
    ids=["pandas", "spark"],
)
def test_unpack_params_with_df(request, df_fixture, transform_func, expected):
    df = request.getfixturevalue(df_fixture)
    result_arg, result_kwarg = _unpack_params(unpack={"df": df, "x": 1})
    result = transform_func(*result_arg, **result_kwarg)
    assert expected(result)


def test_unpack_params_true_decorator():
    result = my_func(unpack={"x": {"c1": 1}})
    assert result == 1


def test_unpack_params_false_decorator():
    with pytest.raises(KeyError, match="c1"):
        my_func({"x": {"c1": 1}})


@make_list_regexable(source_df="df", make_regexable_kwarg="columns")
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
