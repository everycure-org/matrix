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
"""Decorator to unpack parameters into the function."""
import functools

UNPACK_KW = "unpack"


def _unpack_params(*args, **kwargs):
    """Unpacks top level dictionaries in args and kwargs by 1 level.

    Most beneficial if used as part of a kedro node. In cases where we need to pass
    lots of parameters to function, we can use this unpack decorator where we use
    unpack arg or kwarg which will contain most of the parameters.

    To enable unpacking, the keyword "unpack" can be provided via a dictionary in
    args and/or kwargs.
    """
    unpacked_kwargs_from_args = {}
    args = list(args)
    remove_list = []
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            if UNPACK_KW in arg.keys():
                remove_list.append(i)
                new_kwargs = args[i]
                unpacked_kwargs_from_args = {
                    **unpacked_kwargs_from_args,
                    **new_kwargs.pop(UNPACK_KW),
                    **new_kwargs,
                }
    args = [i for j, i in enumerate(args) if j not in remove_list]

    if UNPACK_KW in kwargs:
        unpack_kwargs_from_kwargs = kwargs.pop(UNPACK_KW, {})
    else:
        unpack_kwargs_from_kwargs = {}

    return args, {**unpacked_kwargs_from_args, **kwargs, **unpack_kwargs_from_kwargs}


def unpack_params():  # pylint: disable=missing-return-type-doc
    """Unpack params decorator.

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _unpack_params(*args, **kwargs)
            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorate
