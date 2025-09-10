# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
"""Functions to generate fake data declaratively."""
# pylint: disable=too-many-lines

import datetime
import hashlib
import importlib
import itertools
import logging
import random
import re
import json
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from faker import Faker
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


_DEFAULT_OBJ_PATH = __name__


def probability_null(  # pylint: disable = bad-option-value
    prob_null: float = None,
    null_value: Any = None,
    seed: int = None,
):
    """Inject ``None`` into a list based on a probability.

    Args:
        prob_null: A number between 0 and 1.
        null_value: The value to use for NULLs. Defaults to ``None``.
        seed: For deterministic runs. Defaults to None.

    Returns:
        Wrapper function.

    Raises:
        AssertionError: if ``prob_null`` is not between 0 and 1 (inclusive).
    """

    def decorate(func):
        @wraps(func)
        def wrapper(
            *args, prob_null=prob_null, null_value=null_value, seed=seed, **kwargs
        ):
            if seed:
                random.seed(seed)

            result = func(*args, **kwargs)

            if prob_null:
                assert 0 <= prob_null <= 1, "``prob_null`` should be between 0 and 1!"
                result = [
                    x if prob_null < random.random() else null_value for x in result
                ]

            return result

        return wrapper

    return decorate


@probability_null()
def generate_unique_id(
    num_rows: int,
    prefix: str = "",
    id_start_range: int = 1,
    id_end_range: int = None,
    id_length: int = None,
) -> List[str]:
    """Generate a list of unique integer ids with ``prefix`` prefix.

    Args:
        num_rows: Number of rows to generate
        prefix: prefix to use for your ids. Defaults to empty string "".
        id_start_range: The starting range of the id. Defaults to 1.
        id_end_range: The ending range of the id. Defaults to ``num_rows``.
        id_length: The length of the id. Pads "0" to the left, i.e. an id ``1``, with
            ``id_length`` gives ``0001``.

    Returns:
        A list of generated ids.

    Raises:
        ValueError: if ``id_length`` is lower than the length of the ``prefix`` combined
            with the length of ``id_end_range``
    """
    id_end_range = id_end_range or (num_rows + id_start_range)

    if id_length:
        if len(str(id_end_range)) > (id_length - len(prefix)):
            raise ValueError(
                "The id_length is lower then the prefix combined with the id_end_range."
            )

    unique_numeric_ids = [str(x) for x in range(id_start_range, id_end_range)]

    if id_length:
        unique_numeric_ids = [
            x.zfill(id_length - len(prefix)) for x in unique_numeric_ids
        ]

    unique_ids = [f"{prefix}{x}" for x in unique_numeric_ids]

    if len(unique_ids) == num_rows:
        return unique_ids

    # sample without replacement
    if len(unique_ids) > num_rows:
        sample_ids = random.sample(unique_ids, num_rows)
        sample_ids.sort()
        return sample_ids

    sampled_ids = generate_values(sample_values=unique_ids, num_rows=num_rows)
    return sampled_ids


@probability_null()
def numpy_random(
    num_rows: int, distribution: str, numpy_seed: int = None, **kwargs
) -> List[Any]:
    """Wrapper for numpy.random.

    Generate a column with size `num_rows`
        based on the selected `distribution` and
        numpy seed. It is usefull to take a look on
        https://numpy.org/doc/1.16/reference/routines.random.html
        to know about the distributions and their arguments.

    Args:
        num_rows: Number of rows that will be created.
        distribution: Name of the selected distribution (Eg: Normal).
        numpy_seed: Numpy seed for always getting the same output.
        kwargs: Arguments that are going to be passed to the
            selected distribution.

    Returns:
        column: Calculated column with the selected `distribution`
            and `numpy_seed`.

    Examples:
        Input:
        num_rows= 4
        distribution = "normal"
        mean = 0
        scale = 1
        Output:
        index | value
        0    -0.76909313
        1    0.76632931
        2    -0.23859875
        3    1.23073128
        4    -0.13900229
    """
    if numpy_seed:
        np.random.seed(numpy_seed)

    dist = getattr(np.random, distribution)
    samples = dist(size=num_rows, **kwargs).tolist()
    return samples


@probability_null()
def faker(
    num_rows: int,
    provider: str,
    localisation: Union[str, List[str]] = None,
    provider_args: Dict[str, Union[str, int]] = None,
    faker_seed: int = None,
) -> List[Any]:
    """Thin wrapper for accessing Faker properties.

    Args:
        num_rows: Number of rows that will be created.
        provider: The type of data that you will create. Eg: currency,
        color, bank, address, phone_number
        localisation: The country that you want the data to be generated.
            https://faker.readthedocs.io/en/stable/locales.html for more details.
            Eg : "en_US" (get data from US), ["en_US","ja_JP"] (get data
            from US and Japan).
        provider_args: Dictionary or string of the functions and values that
            are going to be used. Eg: "zipcode"
        faker_seed: Seed to be used by faker. Default : None

    Returns:
        column: Calculated column with the selected `localisation`
            and selected functions by `provider_args`.

    Examples:
        Input:
        num_rows= 4
        distribution = "normal"
        mean = 0
        scale = 1
        Output:
        index | value
        0    -0.76909313
        1    0.76632931
        2    -0.23859875
        3    1.23073128
        4    -0.13900229
    """
    faker_obj = Faker(localisation)
    if faker_seed:
        faker_obj.seed_instance(faker_seed)

    provider_args = provider_args or {}
    faker_results = [
        getattr(faker_obj, provider)(**provider_args) for _ in range(num_rows)
    ]

    return faker_results


@probability_null()
def generate_random_arrays(
    num_rows: int,
    sample_values: List[Any],
    allow_duplicates: bool = False,
    to_json: bool = False,
    delimiter: str = None,
    length: int = None,
) -> List[List[Any]]:
    """Generate random array with sample values.

    Generate a column with size `num_rows`
        based on an array `sample values` with array values
        with length `length`.

    Args:
        num_rows: Number of rows that will be created.
        sample_values: Array that contain the possible values.
        allow_duplicates: Allow duplicated data.
        length: Size of the resulting array (exact).
        to_json: Serialize array as json
        delimiter: Delimiter, if set used to delimit elements in stringified notation.

    Returns:
        column: Calculated column.

    Examples:
        Input:
        num_rows: 4
        sample_values: [1, 2, 3, 4]
        allow_duplicates: False
        length = 2
        Output:
        index | value
        0    [2, 3]
        1    [2, 4]
        2    [3, 1]
        3    [4, 3]
    """
    counts = [1 for _ in range(len(sample_values))]
    if allow_duplicates:
        counts = [random.randint(1, 100) for _ in range(len(sample_values))]

    sample_values_updated = [[x] * i for x, i in zip(sample_values, counts)]
    # Flatten the list
    sample_values_updated = [
        item for sublist in sample_values_updated for item in sublist
    ]

    if length:
        list_of_lists = [
            random.sample(sample_values_updated, k=length) for _ in range(num_rows)
        ]
        return list_of_lists

    res = [
        random.sample(sample_values_updated, k=random.randint(1, len(sample_values)))
        for _ in range(num_rows)
    ]

    if delimiter:
        res = [delimiter.join(row) for row in res]

    if to_json:
        res = [json.dumps(row) for row in res]

    return res

@probability_null()
def generate_random_numbers(
    num_rows: int,
    start_range: int = 0,
    end_range: int = 1,
    integer: bool = False,
) -> List[Union[float, int]]:
    """Generate random numbers between a specified range, of type float or int.

    Args:
        num_rows: Number of rows that will be created.
        start_range: Starting value from the possible values.
            Defaults to 0.
        end_range: Ending value from the possible values.
            Defaults to 1.
        integer: Boolean to define if the output should be an int or
            a float. Defaults to False.

    Returns:
       Generated column of type int or float.

    Examples:
        Input:
        num_rows: 4
        start_range: 3
        end_range: 8
        integer: True
        Output:
        index | value
        0    5
        1    5
        2    3
        3    4
    """
    list_of_random_floats = [
        random.random() * (end_range - start_range) + start_range
        for _ in range(num_rows)
    ]

    if integer:
        return [int(x) for x in list_of_random_floats]

    return list_of_random_floats


@probability_null()
def generate_values_from_weights(
    weights_dict: Dict[str, int], num_rows: int = 1
) -> Any:
    """Generate list of values for the provided weights.

    Args:
        num_rows: Number of rows that will be created.
        weights_dict: Starting value (must be int) from the possible values.

    Returns:
        Column with weighted values.

    Examples:
        Input:
        num_rows: 4
        weights_dict: {"A":"1","B":"3", "C":"9"}
        Output:
        index | value
        0    C
        1    B
        2    C
        3    C
    """
    elements, weights = zip(*weights_dict.items())

    list_of_values = random.choices(elements, weights, k=num_rows)

    return list_of_values


@probability_null()
def generate_values(
    num_rows: int,
    sample_values: Union[List[Any], Dict[str, int]],
    sort_values: bool = True,
) -> List[Any]:
    """Generate a list of values given sample values.

    Args:
        num_rows: Number of rows that will be created.
        sample_values: Can be defined as a list of values or a Dict of
            values and their weights.
        sort_values: Boolean to define if sorting is necessary.

    Returns:
        column: Calculated column of randomly selected.

    Examples:
        Input:
        num_rows: 4(
        sample_values: ["A", "B", "C"]
        Output:
        index | value
        0    B
        1    A
        2    A
        3    C
    """
    if isinstance(sample_values, list):
        weights_dict = {value: 1.0 for value in sample_values}
    elif isinstance(sample_values, dict):
        weights_dict = sample_values
    else:
        raise TypeError(
            "Please provide a list of values or a dictionary of values with weights."
        )

    if len(weights_dict) == num_rows:
        return list(weights_dict.keys())

    # sample without replacement
    if len(weights_dict) > num_rows:
        sample_values = random.sample(list(weights_dict.keys()), num_rows)
        if sort_values:
            sample_values.sort()
        return sample_values

    list_of_values = generate_values_from_weights(
        weights_dict=weights_dict, num_rows=num_rows
    )

    return list_of_values


@probability_null()
def conditional_generate_from_weights(  # pylint: disable=invalid-name
    value: Any, dependent_weights: Dict[Any, Dict[str, int]]
) -> Any:
    """Generates a new distribution conditional on ``value``.

    The distribution is defined in ``dependent_weights``.

    ::

        # given
        dependent_weights = {"a": {'x': 2, 'y': 10}, "b":{'m': 2, 'n': 1, 'p': 10}}

        # and
        v = "a"

        # will return either x or y with a ratio of 2:10

    Args:
        value: The input element.
        dependent_weights: A dictionary of weights for different ``value``.

    Returns:
        A single new element.
    """
    return generate_values_from_weights(weights_dict=dependent_weights[value])[0]


@probability_null()
def conditional_string(
    key: Any, mapping: Dict[Any, Any], null_value: Any = None
) -> Any:
    """Remaps value based on the provided mapping.

    Args:
        key: The input key, typically a cell value in the column.
        mapping: A mapping dictionary.
        null_value: The value to return if key not found. Defaults to ``None``.

    Returns:
        The value based on the input element.
    """
    return mapping.get(key, null_value)


@probability_null()
def hash_string(key: Any, buckets: List[Any]) -> Any:
    """Hashes a value to a given list of buckets.

    The same value will always go to the same bucket.
    Example usage would be: I need to assign 10 unique values that
    might repeat in a column with 100 rows to 5 unique categories, and each unique
    value cannot be placed in different categories. If mapping needs to be known
    ahead of time, consider using ``conditional_string``.

    Note:
        Use with ``row_apply``.

    Args:
        key: Any value, string, or integer.
        buckets: A list of buckets.

    Returns:
        The bucket corresponding to the value.
    """
    key = int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16)
    index = abs(key) % len(buckets)

    return buckets[index]


@probability_null()
def generate_dates(  # pylint: disable=too-many-branches, too-many-arguments
    start_dt: Union[str, datetime.datetime],
    end_dt: Union[str, datetime.datetime],
    freq: str,
    num_rows: int = None,
    sort_dates: bool = True,
    date_format: str = None,
) -> List[datetime.datetime]:
    """Generates a range of dates.

    If ``num_rows`` is specified, it will downsample or upsample accordingly.

    Args:
        start_dt: The start date of the date range.
        end_dt: The end date of the date range.
        freq: Frequency. See:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            # pylint: disable=line-too-long    # noqa: E501
        num_rows: Number of rows to generate
        sort_dates : sort dates. This is useful when creating master records for a
            dimension table. Not all dates/values should be sequential for pseudo-real data
            For example: Join date of clients can be sequential, since their IDs would be
            sequential as well. But the close dates of a list of accounts can be entirely
            random.
        date_format: Used to format the dates according to a specific format. If set,
            the output type of the column corresponds to string.

    Returns:
        A list of dates

    Examples:
        Input:
        num_rows: 4(
        sample_values: ["A", "B", "C"]
        Output:
        index | value
        0    B
        1    A
        2    A
        3    C
    """
    unique_dates = list(pd.date_range(start_dt, end_dt, freq=freq))
    dates = unique_dates

    if num_rows:
        # sample without replacement, downsampling
        if len(unique_dates) > num_rows:
            dates = random.sample(unique_dates, num_rows)

        # upsampling
        if len(unique_dates) < num_rows:
            dates = generate_values(sample_values=unique_dates, num_rows=num_rows)

    # Sort dates
    if sort_dates:
        dates.sort()

    # Format date time
    if date_format:
        dates = [pd.to_datetime(d).strftime(date_format) for d in dates]

    return dates


@probability_null()
def generate_single_date(
    start_dt: Union[str, datetime.datetime],
    end_dt: Union[str, datetime.datetime],
    random_seed: int = None,
) -> List[datetime.datetime]:
    """Generate a single date.

    Args:
        start_dt: The start date of the date range.
        end_dt: The end date of the date range.
        random_seed: The seed for random date generation.

    Returns:
        A single date list

    Examples:
        Input:
        start_dt: 01/01/2022
        end_dt: 01/01/2023
        Output:
        index | value
        0    03/04/2022
    """
    if random_seed:
        random.seed(random_seed)

    if isinstance(start_dt, str):
        start_dt_ts = datetime.datetime.strptime(start_dt, "%Y-%m-%d")
    else:
        start_dt_ts = start_dt

    if isinstance(end_dt, str):
        end_dt_ts = datetime.datetime.strptime(end_dt, "%Y-%m-%d")
    else:
        end_dt_ts = end_dt

    assert start_dt_ts <= end_dt_ts, "``start_dt`` must be less or equal to ``end_dt``!"

    result = random.randrange(start_dt_ts.timestamp(), end_dt_ts.timestamp())
    result = datetime.datetime.fromtimestamp(result).strftime("%Y-%m-%d")

    return [result]


@probability_null()
def drop_duplicates(*list_of_lists: List[List[Any]], position: int = 0) -> List[Any]:
    """Given a list of lists of all same length, returns unique combinations.

    Row ordering is preserved.

    ::

        # given
        [
            [1,2,2,3,3],
            [1,2,2,3,4],
        ]

        # intermediate result
        intermediate_result = [
            (1,1),
            (2,2),
            (3,3),
            (3,4),
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 1 returns the first "column"
        # returns [1,2,3,3]
        # position == 2 returns the second "column"
        # returns [2,3,3,4]

    Args:
        list_of_lists: A list of list of values of equal length.
        position: The n-th position value to return.

    Returns:
        A new list of values.

    Raises:
        ValueError: If lists in main list are not of same length.

    Example1:
        Input:
        list1=[1,4,5,5,5,6,7,7]
        Output:
        index | value
        0 1
        1 4
        2 5
        3 6
        4 7
    Example2:
        Input:
        list1: [3,3,3,4,5,5,6,8]
        list2: [3,3,3,4,5,5,6,9]
        position: 0
        Output:
        index | value
        0 3
        1 4
        2 5
        3 8

        Input:
        list1: [3,3,3,4,5,5,6,8]
        list2: [3,3,3,4,5,5,6,9]
        position: 1
        Output:
        index | value
        0 3
        1 4
        2 5
        3 9
    """
    # check if all inputs length
    expected_length = len(list_of_lists[0])
    for vector in list_of_lists:
        if len(vector) != expected_length:
            raise ValueError("All vectors in list should of same length.")

    results = []

    for tpl in zip(*list_of_lists):
        if tpl not in results:
            results.append(tpl)

    return [tpl[position] for tpl in results]


@probability_null()
def explode(  # pylint: disable=too-many-branches, too-many-arguments, too-many-locals # noqa: C901, E501
    list_of_values: List[List[Any]],
    position: int,
    explode_func: Union[str, Callable],
    explode_func_kwargs: Dict[Any, Any] = None,
    distribution_kwargs: Dict[str, Any] = None,
    num_rows: int = None,
) -> List[Any]:
    """Runs a function for each row in ``list_of_values``.

    ::

        # Given
        l1 = [1, 2]
        l2 = [a, b]
        list_of_values = [l1, l2]
        explode_func: lambda x: [x, y, z]

        # Intermediate result
        intermediate_result = [
            (1, a, x),
            (1, a, y),
            (1, a, z),
            (2, b, x),
            (2, b, y),
            (2, b, z)
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 0
        # returns [1, 1, 1, 2, 2, 2]
        # position == 1
        # returns [a, a, a, b, b, b]
        # position == 2
        # returns [x, y, z, x, y, z]


    Args:
        list_of_values: List of lists that you want to explode. The number of final
            rows will be the product(len(lists)).
        position: Select the index of the resulting tuples.
            Eg: If you pass 3 lists, you can select from which list you want
            the values. The index (position=0), the first column (position=1)
            etc.
        explode_func: Exploding function that you want to be used.
            Eg: `generate_dates`, `lamba x : x`
        explode_func_kwargs: Dictionary of the arguments that are going to be
            passed to the exploding function. Defaults to None.
            Eg: { "start_dt": "2019-01-01", "end_dt": "2020-01-01", "freq": "M",},
        distribution_kwargs: Dictionary of the arguments that are going to be
            passed to the distribution functional. Defaults to None.
            Eg: {"distribution": "gamma", "scale": 10, "shape": 1,},
        num_rows: Exact number of rows that will be created.

    Returns:
        A list of new values.

    Raises:
        ValueError: if output does not equal num_rows if num_rows is specified.

    Example1:
        input:
        list_of_values: [[1,2,3],["a", "b", "c"]]
        position: 3
        explode_func: "lambda x : x"
        explode_func_kwargs: {}
        distribution_kwargs: {}
        num_rows: 6
        Output:
        index | value
        0 1
        1 4
        2 5
        3 6
        4 7
    """
    results = []

    if isinstance(explode_func, str):
        if explode_func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(explode_func)
        else:
            callable_to_apply = load_obj(explode_func)
    else:
        callable_to_apply = explode_func

    if not explode_func_kwargs:
        explode_func_kwargs = {}

    explode_func_kwargs_parsed = deepcopy(explode_func_kwargs)

    # pylint: disable=eval-used
    if distribution_kwargs:
        num_rows_per_row = numpy_random(
            num_rows=len(list_of_values[0]), **distribution_kwargs
        )
        for tpl, n_rows in zip(zip(*list_of_values), num_rows_per_row):
            for key, value in explode_func_kwargs.items():
                if isinstance(value, str) and value.startswith("list_of_values"):
                    list_expr = value.replace("list_of_values", "tpl")
                    explode_func_kwargs_parsed[key] = eval(list_expr)

            tpl_results = callable_to_apply(
                num_rows=int(n_rows), **explode_func_kwargs_parsed
            )

            for row in tpl_results:
                results.append((*tpl, row))
    else:
        for tpl in zip(*list_of_values):
            for key, value in explode_func_kwargs.items():
                if isinstance(value, str) and value.startswith("list_of_values"):
                    list_expr = value.replace("list_of_values", "tpl")
                    explode_func_kwargs_parsed[key] = eval(list_expr)

            tpl_results = callable_to_apply(**explode_func_kwargs_parsed)

            for row in tpl_results:
                results.append((*tpl, row))

    if num_rows:
        if len(results) != num_rows:
            raise ValueError("Ensure function returns ``num_rows``! ")

    return [tpl[position] for tpl in results]


@probability_null()
def drop_filtered_condition_rows(
    *list_of_lists: List[List[Any]], position: int = 0
) -> List[Any]:
    """Given a list of lists of all same length, returns desirable rows.

    ::

        # Given
        l1 = [1,2,3]
        l2 = [x, y, z]
        l3 = [True, True, False]
        list_of_values = [l1, l2, l3]

        # intermediate_result
        intermediate_result = [
            (1, x, True),
            (2, y, True),
            (3, z, False),
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 1 returns the first "column" which are "True" records
        # returns [1,2]
        # position == 2 returns the second "column" which are "True" records
        # returns [x, y]

    Args:
        list_of_lists: A list of list of values of equal length.
        position: The n-th position value to return.

    Returns:
        A new list of values.

    Raises:
        ValueError: if input list is not greater than two vectors and
        If lists in main list are not of same length.
    """
    if len(list_of_lists) < 2:
        raise ValueError("Greater than two vectors expected.")

    expected_length = len(list_of_lists[0])

    for vector in list_of_lists:
        if len(vector) != expected_length:
            raise ValueError("All vectors in list should of same length.")

    results = []

    for tpl in zip(*list_of_lists):
        if True in tpl:
            results.append(tpl)

    return [tpl[position] for tpl in results]


@probability_null()
def cross_product(*list_of_lists: List[Any], position: int = 0) -> List[Any]:
    """Given a list of lists, returns all possible combinations values.

    ::

        # Given
        l1 = [p1,p2]
        l2 = [q1,q2]
        list_of_values = [l1, l2]

        # intermediate_result
        intermediate_result = [
            (p1, q1),
            (p1, q2),
            (p2, q1),
            (p2, q2),
        ]

        # return [tpl[position] for tpl in intermediate_result]

        # position == 0 returns the first "column"
        # returns [p1, p1, p2, p2]
        # position == 1 returns the second "column"
        # returns [q1, q1, q2, q2]

    Args:
        list_of_lists: A list of list
        position: The n-th position value to return.

    Returns:
        A new list of values.

    Raises:
        If position is out of range.
    """
    if position >= len(list_of_lists) or position < 0:
        raise ValueError(
            f"Position out of range. Should vary between 0 "
            f"and {len(list_of_lists) - 1}, but received {position}."
        )

    results = list(itertools.product(*list_of_lists))
    return [tpl[position] for tpl in results]


@probability_null()
def column_apply(  # pylint: disable=too-many-branches # noqa: C901
    list_of_values: Union[List[Any], List[List[Any]]],
    column_func: Union[str, Callable],
    column_func_kwargs: Dict[str, str] = None,
    num_rows: int = None,
    resize: bool = False,
    check_all_inputs_same_length: bool = True,
) -> List[Any]:
    """Generic wrapper to call a function on a list.

    Function does not modify order of elements in list unless resize is True.

    ::

        # Given
        l1 = [1,2,3]
        l2 = [2,3,4]
        list_of_values = [l1, l2]
        column_function = lambda x,y: [(v1, v2) for v1, v2 in zip(x,y)]

        # internally entire column(s) is fed into function
        intermediate_result = column_function(*list_of_values, **column_function_kwargs)

        # intermediate_result
        intermediate_result = [
            (1, 2),
            (2, 3),
            (3, 4),
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 1 returns the first "column"
        # returns [1,2,3]
        # position == 2 returns the second "column"
        # returns [2,3,4]


    Args:
        list_of_values: A list of values or a list containing lists of values.
        column_func: The function to be applied to the list of columns.
        column_func_kwargs: Additional kwargs.
        num_rows: Checks output length if specified. Defaults to None.
        resize: Whether to resize the result. Defaults to False.
        check_all_inputs_same_length: Whether to bypass the same length check. Defaults
            to False. Used mainly with `cross_product`.

    Returns:
        A list of values

    Raises:
        ValueError: If vectors in list are not of same length or if
            function does not return the expected number of rows.

    Example1:
        input:
        list_of_values: [[1,2,3],["a", "b", "c"]]
        column_func: "lambda x,y : x"
        column_func_kwargs: {}
        num_rows: 3
        resize: ""
        check_all_inputs_same_length: ""
        Output:
        index | value
        0 1
        1 2
        2 3
        ...
    """
    if isinstance(column_func, str):
        if column_func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(column_func)
        else:
            callable_to_apply = load_obj(column_func)
    else:
        callable_to_apply = column_func

    if not isinstance(list_of_values[0], list):
        list_of_values = [list_of_values]

    if not column_func_kwargs:
        column_func_kwargs = {}

    # check if all inputs length
    expected_length = len(list_of_values[0])
    for vector in list_of_values:
        if len(vector) != expected_length:
            if check_all_inputs_same_length:
                raise ValueError("All vectors in list should of same length.")

    results = callable_to_apply(*list_of_values, **column_func_kwargs)

    if resize:
        if len(results) != num_rows:
            logger.warning("Resizing list from %s to %s", len(results), num_rows)

            # sample without replacement
            if len(results) > num_rows:
                results = random.sample(results, num_rows)

            else:
                results = generate_values(num_rows=num_rows, sample_values=results)

    if num_rows:
        if len(results) != num_rows:
            raise ValueError("Ensure function returns ``num_rows``! ")

    return results


@probability_null()
def row_apply(  # pylint: disable=too-many-branches # noqa: C901
    list_of_values: Union[List[Any], List[List[Any]]],
    row_func: Union[str, Callable],
    row_func_kwargs: Dict[str, str] = None,
    num_rows: int = None,
    resize: bool = False,
) -> List[any]:
    """Generic wrapper to call a function for every element in a list.

    Function does not modify order of elements in list, unless resize is true.

    ::

        l1 = [1,2,3]
        l2 = [2,3,4]
        list_of_values = [l1, l2]
        row_func: lambda x, y: x+y

        # row function is applied element wise
        intermediate_result = [row_function(*tpl) for tpl in zip(*list_of_values)]

        # returns
        [
            1+2,
            2+3,
            3+4,
        ]


    Args:
        list_of_values: The list containing elements.
        row_func: The function to be applied to each element.
        row_func_kwargs: Additional kwargs.
        num_rows: Checks output length if specified. Defaults to None.
        resize: Whether to resize the result. Defaults to False.

    Returns:
        A new list of values

    Raises:
        ValueError: if lists in `list_of_values` are not of same length.
    """
    if isinstance(row_func, str):
        if row_func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(row_func)
        else:
            callable_to_apply = load_obj(row_func)
    else:
        callable_to_apply = row_func

    if not isinstance(list_of_values[0], list):
        list_of_values = [list_of_values]

    if not row_func_kwargs:
        row_func_kwargs = {}

    # check if all inputs are of same length
    expected_length = len(list_of_values[0])
    for vector in list_of_values:
        if len(vector) != expected_length:
            raise ValueError("All vectors in list should of same length.")

    # pylint: disable=unnecessary-comprehension
    # convert to common format
    list_of_tuples = [tpl for tpl in zip(*list_of_values)]
    # pylint: enable=unnecessary-comprehension

    if resize:
        if len(list_of_tuples) != num_rows:
            logger.warning("Resizing list from %s to %s", len(list_of_tuples), num_rows)

            # sample without replacement
            if len(list_of_tuples) > num_rows:
                list_of_tuples = random.sample(list_of_tuples, num_rows)

            else:
                list_of_tuples = generate_values(
                    num_rows=num_rows, sample_values=list_of_tuples
                )

    results = [callable_to_apply(*tpl, **row_func_kwargs) for tpl in list_of_tuples]

    if num_rows:
        if len(results) != num_rows:
            raise ValueError("Ensure function returns ``num_rows``! ")

    return results


class MockDataGenerator:
    """Class to handle a set of dataframe fabrication instructions."""

    def __init__(self, instructions: Dict[str, Any], seed: int = None):
        """Init class."""
        self.all_instructions = instructions

        self._num_rows = None
        self.all_dataframes = {}
        self.seed = seed

        # set all possible known seeds here
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate_all(self):
        """Run the generator for all dataframes."""
        for df_name, df_instructions in self.all_instructions.items():
            self.generate_dataframe(df_name=df_name, df_instructions=df_instructions)

    def generate_dataframe(self, df_name: str, df_instructions: Dict[str, Any]):
        """Generates a dataframe given a dictionary of column instructions.

        Args:
            df_name: The name of the dataframe being generated.
            df_instructions: The set of column instructions.
        """
        logger.info("Generating for df: %s", df_name)

        self._num_rows = df_instructions.pop("num_rows", None)

        # pylint: disable=simplifiable-if-statement,invalid-name
        if not self._num_rows:
            infer_num_rows_from_first_column = True
        else:
            infer_num_rows_from_first_column = False

        df_column_instructions = df_instructions["columns"]

        temp_results = {}

        for column_name, column_instructions in df_column_instructions.items():
            logger.info("Generating column: %s", column_name)
            column_instructions["num_rows"] = self._num_rows

            temp_results[column_name] = self.generate_column(column_instructions)

            # infer num_rows from 1st column
            if infer_num_rows_from_first_column:
                self._num_rows = len(temp_results[column_name])
                logger.info(
                    "Inferring number of rows from %s.%s: %s",
                    df_name,
                    column_name,
                    self._num_rows,
                )
                # execute only once per dataframe
                infer_num_rows_from_first_column = False

            # update this object as we go along
            # so that any column built is already immediately available
            self.all_dataframes[df_name] = pd.DataFrame(temp_results).copy(deep=True)

    def generate_column(
        self,
        column_instructions: Dict[str, Any],
    ) -> pd.array:
        """Generates a column given instructions.

        Args:
            column_instructions: Dictionary of column instructions.

        Returns:
            None
        """
        logger.info("column_instructions: %s", column_instructions)

        func_type = column_instructions.pop("type")

        # Get column values from list_of_values
        if func_type in ["row_apply", "column_apply", "explode"]:
            potential_string_references = column_instructions["list_of_values"]
            actual_values = self.parse_column_reference(potential_string_references)
            column_instructions["list_of_values"] = actual_values

        # Get column values from sample_values in explode to generate_values function
        if (
            func_type in ["explode"]
            and column_instructions["explode_func"] == "generate_values"
        ):
            potential_string_references = column_instructions["explode_func_kwargs"][
                "sample_values"
            ]
            actual_values = self.parse_column_reference(potential_string_references)
            # Grab inner list if values come from column reference
            if len(actual_values) == 1 and isinstance(actual_values[0], list):
                column_instructions["explode_func_kwargs"][
                    "sample_values"
                ] = actual_values[0]
            else:
                column_instructions["explode_func_kwargs"][
                    "sample_values"
                ] = actual_values

        # Get column values from sample_values in generate_values function
        if func_type in ["generate_values"]:
            potential_string_references = column_instructions["sample_values"]
            actual_values = self.parse_column_reference(potential_string_references)
            # Grab inner list if values come from column reference
            if len(actual_values) == 1 and isinstance(actual_values[0], list):
                column_instructions["sample_values"] = actual_values[0]
            else:
                column_instructions["sample_values"] = actual_values

        func_to_apply = load_obj(
            func_type,
        )

        if column_instructions.get("integer") not in [True, False, None]:
            raise ValueError(
                f"Argument 'integer' must be 'True', 'False' or empty. It can't be "
                f"{column_instructions.get('integer')}"
            )
        if column_instructions.get("integer"):
            if column_instructions.get("dtype", None):
                dtype = column_instructions.get("dtype", None)
                raise ValueError(
                    f"Argument 'integer' cannot be set with dtype: " f"{dtype}"
                )
            dtype = "Int64"
        else:
            # For data types allowed,
            # see: https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
            dtype = column_instructions.pop("dtype", None)

        results = func_to_apply(**column_instructions)
        return pd.array(results, dtype=dtype)

    def _parse_column_reference(self, string_reference: str) -> List[Any]:
        """Parses column reference(s) and returns the actual column.

        Assumes column has already been generated earlier.

        Args:
            string_reference: A string ``df.column_name``.

        Returns:
            A new list containing the actual values from the column.
        """
        reference = string_reference.split(".")
        df_name = reference[0]
        column_name = reference[1]
        # extract list of dataframe column
        # if column is datetime, a different logic is applied to ensure that datetime
        # objects are returned
        if is_datetime64_any_dtype(self.all_dataframes[df_name][column_name]):
            temp_df = self.all_dataframes[df_name].copy(deep=True)
            actual_values = pd.to_datetime(temp_df[column_name].to_numpy()).tolist()
        else:
            actual_values = np.array(
                self.all_dataframes[df_name][column_name].values
            ).tolist()

        return actual_values

    def parse_column_reference(
        self, string_references: Union[str, List[Any], Dict[str, int]]
    ) -> Union[List[Any], Dict[str, int]]:
        """Parses sample values, and processes them if they are column references.

        Args:
            string_references: A string reference to a pre-made column,
                a list of values, or distribution of values.

        Returns:
            A new list containing the actual values from the column if column reference
            was passed.
        """
        if (
            isinstance(string_references, list)
            and isinstance(string_references[0], str)
            # match table.name or tab_le.na_me or _Tab_le.name or _TAB_le1.Nam
            and re.match("^[A-Za-z0-9_]+\\.[A-Za-z0-9_:]+$", string_references[0])
        ):
            actual_values = [
                self._parse_column_reference(string_reference)
                for string_reference in string_references
            ]
        elif isinstance(string_references, str) and "." in string_references:
            actual_values = [self._parse_column_reference(string_references)]
        else:
            actual_values = string_references
        return actual_values


def load_obj(
    obj_path: str,
    default_obj_path: str = _DEFAULT_OBJ_PATH,
) -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(  # pylint: disable=consider-using-f-string, line-too-long # noqa: E501
                obj_name, obj_path
            )
        )
    return getattr(module_obj, obj_name)


def load_callable_with_libraries(function: str) -> Callable:
    """Evaluates a string function to convert into a callable.

    Imports libraries if any are used.

    Args:
        function: string function to convert to callable

    Returns:
        Callable of the input function
    """
    if "." in function:
        libraries_to_import = [
            element.split(".")[0]
            for element in re.split(r"[^a-zA-Z.]", function)
            if re.match(r"[a-z]{2,}\.[a-z]+", element)
        ]

        for library in libraries_to_import:
            globals()[library] = __import__(library)

    callable_func = eval(function)  # pylint: disable=eval-used

    return callable_func
