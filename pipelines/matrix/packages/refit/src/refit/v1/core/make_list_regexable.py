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
# pylint: disable=missing-return-type-doc
"""Decorator to enable regex selection of columns from a dataframe."""
import logging
import re
from inspect import getfullargspec
from typing import List

from boltons.funcutils import wraps

logger = logging.getLogger(__file__)

ENABLE_REGEXABLE_KWARG = "enable_regex"


def _extract_elements_in_list(
    full_list_of_columns: List[str], list_of_regexes: List[str], raise_exc,
) -> List[str]:
    """Use regex to extract elements in a list."""
    results = []
    for regex in list_of_regexes:
        matches = list(filter(re.compile(regex).match, full_list_of_columns))
        if matches:
            for match in matches:
                if match not in results:
                    logger.info("The regex %s matched %s.", regex, match)
                    results.append(  # helps keep relative ordering as defined in YAML
                        match
                    )
        else:
            if raise_exc:
                raise ValueError(
                    f"The following regex did not return a result: {regex}."
                )
            logger.warning("The following regex did not return a result: %s", regex)
    return results


def make_list_regexable(
    source_df: str = None, make_regexable: str = None, raise_exc: bool = False,
):
    """Allow processing of regex in input list.

    Args:
        source_df: Name of the dataframe containing actual list columns names.
        make_regexable: Name of list with regexes.
        raise_exc: Whether to raise an exception or just log the warning.
           Defaults to False.

    Returns:
        A wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(
            *args,
            source_df=source_df,
            make_regexable=make_regexable,
            raise_exc=raise_exc,
            **kwargs,
        ):

            argspec = getfullargspec(func)

            enable_regex_index = argspec.args.index(ENABLE_REGEXABLE_KWARG)

            enable_regex = args[enable_regex_index]
            if enable_regex:
                if source_df not in argspec.args:
                    raise ValueError("Please provide source dataframe.")

                if (make_regexable is not None) and (make_regexable in argspec.args):

                    df_index = argspec.args.index(source_df)
                    list_index = argspec.args.index(make_regexable)

                    df = args[df_index]
                    make_regexable_list = args[list_index]

                    if make_regexable_list is not None:
                        df_columns = df.columns
                        new_columns = _extract_elements_in_list(
                            full_list_of_columns=df_columns,
                            list_of_regexes=make_regexable_list,
                            raise_exc=raise_exc,
                        )
                        args = [
                            (new_columns if i == list_index else arg)
                            for (i, arg) in enumerate(args)
                        ]

            result_df = func(*args, **kwargs)

            return result_df

        return _wrapper

    return _decorate
