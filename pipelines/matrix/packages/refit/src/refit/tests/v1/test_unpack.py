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


import pyspark.sql.functions as f
import pytest

from refit.v1.core.unpack import _unpack_params, unpack_params


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


def test_unpack_params_with_spark_df(dummy_spark_df):
    result_arg, result_kwarg = _unpack_params(unpack={"df": dummy_spark_df, "x": 1})

    def dummy_func(df, x):
        df = df.withColumn("new", f.lit(x) + 1)
        return df

    result = dummy_func(*result_arg, **result_kwarg)
    assert [x.asDict() for x in result.select("new").collect()] == [{"new": 2}]


class TestUnpackParams:
    def test_unpack_params_true_decorator(self):
        result = my_func(unpack={"x": {"c1": 1}})
        assert result == 1

    def test_unpack_params_false_decorator(self):
        with pytest.raises(KeyError, match="c1"):
            my_func({"x": {"c1": 1}})
