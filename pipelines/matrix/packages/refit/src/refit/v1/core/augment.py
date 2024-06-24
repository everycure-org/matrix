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
"""Contains the augment decorator."""

import functools

from refit.v1.core.fill_nulls import _fill_nulls
from refit.v1.core.has_schema import _input_has_schema_node, _output_has_schema_node
from refit.v1.core.inject import _inject_object
from refit.v1.core.input_kwarg_filter import _add_input_kwarg_filter
from refit.v1.core.input_kwarg_select import _add_input_kwarg_select
from refit.v1.core.output_filter import _add_output_filter
from refit.v1.core.retry import _retry
from refit.v1.core.unpack import _unpack_params


def augment():
    """Adds additional node functionalities to any function."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            args, kwargs = _add_input_kwarg_filter(*args, **kwargs)
            args, kwargs = _add_input_kwarg_select(*args, **kwargs)
            args, kwargs = _input_has_schema_node(*args, **kwargs)
            args, kwargs = _unpack_params(*args, **kwargs)
            args, kwargs = _inject_object(*args, **kwargs)
            new_func, args, kwargs = _retry(func, *args, **kwargs)
            new_func, args, kwargs = _add_output_filter(new_func, *args, **kwargs)
            new_func, args, kwargs = _fill_nulls(new_func, *args, **kwargs)
            new_func, args, kwargs = _output_has_schema_node(new_func, *args, **kwargs)
            result_df = new_func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate
