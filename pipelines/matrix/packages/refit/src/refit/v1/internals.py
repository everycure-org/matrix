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
"""Internal functions."""
DEFAULT_BEHAVIOUR = False


def _get_param_from_arg_and_kwarg(keyword: str, *args, **kwargs):
    """Parse args and kwargs for certain keywords."""
    kw_in_kwargs = kwargs.pop(keyword, DEFAULT_BEHAVIOUR)

    kw_in_args = DEFAULT_BEHAVIOUR
    for arg in args:
        if isinstance(arg, dict):
            if keyword in arg.keys():
                kw_in_args = arg.pop(keyword)

    kw = kw_in_kwargs or kw_in_args  # pylint: disable=invalid-name

    return kw, args, kwargs
