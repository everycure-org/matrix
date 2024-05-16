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

from refit.tests.v1.conftest import CustomException, dummy_func
from refit.v1.core.retry import _retry, retry


def dummy_raise_value_error(x):
    raise ValueError("Found ValueError.")


def dummy_raise_type_error(x):
    raise TypeError("Found TypeError.")


def dummy_raise_not_implemented_error(x):
    raise NotImplementedError("Found NotImplemented.")


def dummy_raise_custom_exception(x):
    raise CustomException(x)


@retry()
def dummy_func_raises_dec(*args, **kwargs):
    def dummy_raises(x):
        raise TypeError("random TypeError")

    return dummy_raises(*args, **kwargs)


def test_retry_no_error():
    new_func, args, kwargs = _retry(
        dummy_func,
        **{
            "retry": {"exception": [ValueError], "max_tries": 5, "interval": 0.01,},
            "x": 2,
        }
    )

    result = new_func(*args, **kwargs)
    assert result == 2


def test_retry_basic(caplog):
    new_func, args, kwargs = _retry(
        dummy_raise_value_error,
        **{
            "retry": {"exception": [ValueError], "max_tries": 5, "interval": 0.01,},
            "x": 2,
        }
    )

    try:
        new_func(*args, **kwargs)
    except:
        pass

    assert len(caplog.record_tuples) == 5


def test_retry_more_errors(caplog):
    new_func, args, kwargs = _retry(
        dummy_raise_type_error,
        **{
            "retry": {
                "exception": [ValueError, TypeError],
                "max_tries": 10,
                "interval": 0.01,
            },
            "x": 1,
        }
    )
    try:
        new_func(*args, **kwargs)
    except:
        pass

    assert len(caplog.record_tuples) == 10


def test_retry_uncaught_error():
    new_func, args, kwargs = _retry(
        dummy_raise_not_implemented_error,
        **{
            "retry": {
                "exception": [ValueError, TypeError],
                "max_tries": 10,
                "interval": 0.01,
            },
            "x": 1,
        }
    )
    with pytest.raises(NotImplementedError):
        new_func(*args, **kwargs)


def test_retry_custom_exception(caplog):
    new_func, args, kwargs = _retry(
        dummy_raise_custom_exception,
        **{
            "retry": {
                "exception": [CustomException],
                "max_tries": 5,
                "interval": 0.01,
            },
            "x": 2,
        }
    )

    try:
        new_func(*args, **kwargs)
    except:
        pass

    assert len(caplog.record_tuples) == 5


def test_retry_decorator(caplog):
    try:
        dummy_func_raises_dec(
            **{
                "retry": {"exception": [TypeError], "max_tries": 5, "interval": 0.01,},
                "x": 2,
            }
        )
    except:
        pass

    assert len(caplog.record_tuples) == 5
