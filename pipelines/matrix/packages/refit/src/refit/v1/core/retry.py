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
"""Decorator to retry a function given an exception."""
import functools
import logging
import time

from refit.v1.internals import _get_param_from_arg_and_kwarg

RETRY_KW = "retry"


def _retry(func, *args, **kwargs):
    """Retry a function given a list of exceptions.

    Dictionary format expected
    ::
       {
            "retry": {"exception": [ValueError], "max_tries": 5, "interval": 0.01}
       }

    Where
        exception: A list of exceptions to retry.
        max_tries: Number of times to retry.
        interval: The time in seconds of how long to wait before retrying.
    """
    retry_kw, args, kwargs = _get_param_from_arg_and_kwarg(RETRY_KW, *args, **kwargs)

    if retry_kw:
        max_tries = retry_kw.get("max_tries", 1)
        interval = retry_kw.get("interval", 1)
        exception = retry_kw.get("exception", Exception)

        # pylint: disable=line-too-long
        # https://stackoverflow.com/questions/7273474/behavior-of-pythons-time-sleep0-under-linux-does-it-cause-a-context-switch  # noqa: E501
        # pylint: enable=line-too-long
        if not interval > 0:
            logging.warning(
                "Interval set to 0. It is advised to set interval to greater than 0."
            )

        if not isinstance(exception, list):
            exception = [exception]

        def wrapper(*args, **kwargs):
            i = 0
            while i < max_tries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except tuple(  # pylint: disable=catching-non-exception
                    exception
                ) as caught_exception:
                    i = i + 1
                    logging.error(
                        "Hit exception: %s. Sleeping: %s seconds.",
                        caught_exception,
                        interval,
                    )
                    time.sleep(interval)
                except Exception as e:  # pylint: disable=invalid-name
                    raise e

        return wrapper, args, kwargs

    return func, args, kwargs


def retry():  # pylint: disable=missing-return-type-doc
    """Retry function decorator.

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            new_func, args, kwargs = _retry(func, *args, **kwargs)

            result = new_func(*args, **kwargs)

            return result

        return wrapper

    return decorate
