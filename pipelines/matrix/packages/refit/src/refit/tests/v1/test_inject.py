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
import inspect
from copy import deepcopy
from types import FunctionType

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from refit.tests.v1.conftest import dummy_func
from refit.v1.core.inject import _inject_object, _parse_for_objects, inject_object


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
            "object": "refit.tests.v1.conftest.dummy_func",
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
        "plain_func": {"object": "refit.tests.v1.conftest.dummy_func",},
        "inception": {
            "object": "refit.tests.v1.conftest.dummy_func",
            "x": {"object": "refit.tests.v1.conftest.dummy_func"},
        },
        "list_of_objs": [
            {"object": "refit.tests.v1.conftest.dummy_func"},
            {"object": "refit.tests.v1.conftest.dummy_func"},
            {
                "object": "refit.tests.v1.conftest.dummy_func",
                "x": {"object": "refit.tests.v1.conftest.dummy_func",},
            },
        ],
        "list_of_objs_nested": [
            {"my_func": {"object": "refit.tests.v1.conftest.dummy_func",}}
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
        {"object": "refit.tests.v1.conftest.dummy_func"},
        {"object": "refit.tests.v1.conftest.dummy_func"},
    ]
    return flat_params


@inject_object()
def my_func_pd_imputer(*args, **kwargs):
    def dummy_func(df, x, y, imputer):
        df["new"] = df["c1"] + x + y
        return df, imputer

    return dummy_func(*args, **kwargs)


class TestParseObjects:
    def test_class(self, param):
        result = _parse_for_objects(param)
        assert isinstance(result["my_imputer"], SimpleImputer)

    def test_nested_params(self, nested_params):
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
        assert (
            result["list_of_objs_nested"][0]["my_func"].__name__ == dummy_func.__name__
        )

    def test_flat_params(self, flat_params):
        result = _parse_for_objects(flat_params)
        assert result[0].__name__ == dummy_func.__name__
        assert result[1].__name__ == dummy_func.__name__

    def test_numbers_as_keys(self, number_as_keys):
        new_args, new_kwargs = _inject_object(*number_as_keys)
        assert new_args[0] == number_as_keys[0]

    def test_invalid_keyword(self):
        invalid_param = {"object_invalid": "refit.tests.v1.conftest.dummy_func"}
        result = _parse_for_objects(invalid_param)
        assert isinstance(result["object_invalid"], str)  # it should not load object

    def test_instantiate_function_without_input(self):
        instantiate_param = {
            "object": "refit.tests.v1.conftest.dummy_func_without_input",
            "instantiate": True,
        }
        no_instantiate_param = {
            "object": "refit.tests.v1.conftest.dummy_func_without_input",
            "instantiate": False,
        }
        default_param = {
            "object": "refit.tests.v1.conftest.dummy_func_without_input",
        }

        result1 = _parse_for_objects(instantiate_param)
        result2 = _parse_for_objects(no_instantiate_param)
        result3 = _parse_for_objects(default_param)

        assert result1 == "hello world"
        assert isinstance(result2, FunctionType)
        assert isinstance(result3, FunctionType)

    def test_instantiate_function_with_input(self):
        instantiate_param = {
            "object": "refit.tests.v1.conftest.dummy_func",
            "x": 1,
            "instantiate": True,
        }
        no_instantiate_param = {
            "object": "refit.tests.v1.conftest.dummy_func",
            "instantiate": False,
        }
        default_param = {
            "object": "refit.tests.v1.conftest.dummy_func",
            "x": 1,
        }
        default_param_with_tuple = {
            "object": "refit.tests.v1.conftest.dummy_func",
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

    def test_instantiate_class(self):
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


class TestInjectObject:
    def test_class(self, param):
        new_args, new_kwargs = _inject_object(**param)
        assert not new_args
        assert isinstance(new_kwargs["my_imputer"], SimpleImputer)

    def test_nested_params(self, nested_params):
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
        assert (
            new_kwargs["list_of_objs_nested"][0]["my_func"].__name__
            == dummy_func.__name__
        )
        assert isinstance(new_kwargs["df"], pd.DataFrame)

    def test_exclude_kwargs_as_dict_key(self, nested_params):
        _, new_kwargs = _inject_object(**nested_params, exclude_kwargs=["z"])
        assert "z" in new_kwargs.keys()
        assert "object" in new_kwargs["z"].keys()

    def test_exclude_kwargs_as_params(self, another_param):
        _, new_kwargs = _inject_object(
            **another_param, exclude_kwargs=["cv", "estimator"]
        )

        # test estimator key still is in refit syntax
        tuner_object = new_kwargs["tuner"]
        assert hasattr(tuner_object, "estimator")
        assert "object" in tuner_object.estimator.keys()

        # test cv key still is in refit syntax
        assert hasattr(tuner_object, "cv")
        assert "object" in tuner_object.cv.keys()

    def test_config_is_not_mutated(self, nested_params):
        nested_params_copy = deepcopy(nested_params)
        _inject_object(**nested_params)
        df_copy = nested_params_copy.pop("df")
        df = nested_params.pop("df")
        assert nested_params_copy == nested_params
        pd.testing.assert_frame_equal(df, df_copy)

    def test_flat_params(self, flat_params):
        new_args, new_kwargs = _inject_object(*flat_params)
        assert not new_kwargs
        assert new_args[0].__name__ == dummy_func.__name__
        assert new_args[1].__name__ == dummy_func.__name__

    def test_invalid_keyword(self):
        invalid_param = {"object_invalid": "refit.tests.v1.conftest.dummy_func"}
        new_args, new_kwargs = _inject_object(**invalid_param)
        assert not new_args
        assert isinstance(
            new_kwargs["object_invalid"], str
        )  # it should not load objects

    def test_additional_params(self, another_param):
        new_args, new_kwargs = _inject_object(another_param)
        assert new_args[0]
        assert isinstance(new_args[0]["tuner"], GridSearchCV)


class TestInjectObjectDecorator:
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
