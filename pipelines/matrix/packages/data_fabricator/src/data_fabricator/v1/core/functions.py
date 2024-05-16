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
import itertools
import logging
import random
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from faker import Faker

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
            with the length of ``id_end_range``.
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

    """
    if numpy_seed:
        np.random.seed(numpy_seed)
    # Get the attributes from the selected distribution
    dist = getattr(np.random, distribution)
    # Create a list of the results with size `num_rows`
    # with the passed **kwargs.
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

    Returns:
        column: Calculated column.

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

    return [
        random.sample(sample_values_updated, k=random.randint(1, len(sample_values)))
        for _ in range(num_rows)
    ]


@probability_null()
def generate_random_numbers(
    num_rows: int,
    floored: bool = False,
    start_range: int = 0,
    end_range: int = 1,
    precision: int = None,
) -> List[float]:
    """Generate random numbers between a specified range, of type float or int.

    Args:
        num_rows: Number of rows that will be created.
        floored: Flag determining if the random numbers should be floored to enable
        casting to nullable integers.
        start_range: Starting value (must be int) from the possible values.
            Defaults to 0.
        end_range: Ending value (must be int) from the possible values.
            Defaults to 1.
        precision: nth decimal place to round a float number. Defaults to None.

    Returns:
        column: Calculated column of floats.

    """
    result = [
        random.random() * (end_range - start_range) + start_range
        for _ in range(num_rows)
    ]
    if floored:
        logger.info(
            "Detected nullable integer type."
            "Flooring values to enable casting to int."
        )
        result = np.floor(result)
        return result

    if precision:
        return [round(x, precision) for x in result]
    return result


@probability_null()
def generate_values_from_weights(
    weights_dict: Dict[str, int], num_rows: int = 1
) -> Any:
    """Generate list of values for the provided weights.

    Args:
        num_rows: Number of rows that will be created.
        weights_dict: Starting value (must be int) from the possible values.

    Returns:
        column: Calculated column weighted values.

    """
    elements, weights = zip(*weights_dict.items())
    # This function receives an array of values, an array of weights
    # and a integer with the lenght of the returned list.
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
    """Generate a new distribution conditional on ``value``.

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
    """Remap value based on the provided mapping.

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
    """Hash a value to a given list of buckets.

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
    """Generate a range of dates.

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
    """Given a list of lists of all same length, return unique combinations.

    This algorithm works in 2 different ways.
    1 - If it receives one list, it will remove duplicates from the list.
    2 - If it receives more than one list, it will remove duplicates from
    the tuples made from the same position of the lists. See the example please.

    Args:
        list_of_lists: List of lists that may have duplicated lists inside.
        position: Position of the list that you want the values.
            Eg: If you pass 3 lists, you can select from which list you want
            the values.

    Returns:
        A list without duplicates.

    """
    # check if all inputs length
    expected_length = len(list_of_lists[0])
    for vector in list_of_lists:
        if len(vector) != expected_length:
            raise ValueError("All vectors in list should of same length.")

    results = []

    # create tuples from the same position values from a list
    # and create a list of unique tuples.
    for tpl in zip(*list_of_lists):
        if tpl not in results:
            results.append(tpl)

    return [tpl[position] for tpl in results]


@probability_null()
def join_column(
    selected_column: str,
    this_table_key: Union[str, List[str]],
    other_table_key: Union[str, List[str]],
    column_names: List[str],
    list_of_values: List[any],
):
    """Generate the joined column based on the key relationship.

    Args:
        num_rows: Number of rows that need to be generated for this column

    Returns:
        list of column values mapped through key relationship.
    """
    # build a dataframe with selected columns other table
    df_other = pd.DataFrame(
        {
            f"{k}": v
            for k, v in zip(
                column_names[len(this_table_key) :],  # noqa: E203
                list_of_values[len(this_table_key) :],  # noqa: E203
            )
        }
    )
    # same for current table
    df_this = pd.DataFrame(
        {
            f"{k}": v
            for k, v in zip(
                column_names[0 : len(this_table_key)],  # noqa: E203
                list_of_values[0 : len(this_table_key)],  # noqa: E203
            )
        }
    )
    # merge to  dataframe to get values brought
    result_df = pd.merge(
        df_this,
        df_other,
        how="left",
        left_on=this_table_key,
        right_on=other_table_key,
    )
    result_list = result_df[selected_column].to_list()

    return result_list


@probability_null()
def explode(  # pylint: disable=too-many-branches, too-many-arguments, too-many-locals # noqa: C901, E501
    list_of_values: List[List[Any]],
    position: int,
    explode_func: Union[str, Callable],
    explode_func_kwargs: Dict[Any, Any] = None,
    distribution_kwargs: Dict[str, Any] = None,
    num_rows: int = 0,
) -> List[Any]:
    """Run a function for each row in ``list_of_values``.

    This function explode the values from `list_of_values` in
    the received order by using an explode function. Here you will use
    tuples,

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
            passed to the exploding fucntion.
            Eg: { "start_dt": "2019-01-01", "end_dt": "2020-01-01", "freq": "M",},
        distribution_kwargs: Dictionary of the arguments that are going to be
            passed to the distribution functional.
            Eg: {"distribution": "gamma", "scale": 10, "shape": 1,},
        num_rows: Exact number of rows that will be created.

    Returns:
        A list (column) of values exploded.

    """
    results = []

    callable_to_apply = load_function_if_string(explode_func)

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
def cross_product(*list_of_lists: List[Any], position: int = 0) -> List[Any]:
    """Given a list of lists, return all possible combinations values.

    This function uses a itertools function in order to create all the
    combinations possible between the lists that are passed as the list
    of lists.

    Args:
        list_of_lists: List of lists that you want to combine. The number of final
            rows will be the product(len(lists)).
        position: Select the index of the resulting tuples.
            Eg: If you pass 3 lists, you can select from which list you want
            the values. The index (position=0), the first column (position=1)
            etc.

    Returns:
        A list (column) of values exploded.

        ...
    """
    if position >= len(list_of_lists) or position < 0:
        raise ValueError(
            f"Position out of range. Should vary between 0 "
            f"and {len(list_of_lists) - 1}, but received {position}."
        )

    results = list(itertools.product(*list_of_lists))
    return [tpl[position] for tpl in results]


@probability_null()
def cross_product_with_separator(
    *list_of_lists: List[Any], separator: str = ","
) -> List[Any]:
    """Given a list of lists, return possible combinations values.

    This function uses a itertools function in order to create all the
    combinations possible between the lists that are passed as the list
    of lists and replace the separator from it.

    Args:
        list_of_lists: List of lists that you want to combine. The number of final
            rows will be the product(len(lists)).
            separator: The given separator for the result.
            Eg: " " will return ["blue car", "blue bike", ...].

    Returns:
        A list (column) of values exploded with the designed separator

    """
    results = list(itertools.product(*list_of_lists))
    # Remove the first and the last character (the parenthesis)
    # and the single commas
    return [separator.join(tpl) for tpl in results]


@probability_null()
def column_apply(  # pylint: disable=too-many-branches # noqa: C901
    list_of_values: Union[List[Any], List[List[Any]]],
    column_func: Union[str, Callable],
    column_func_kwargs: Dict[str, str] = None,
    num_rows: int = 0,
    resize: bool = False,
    check_all_inputs_same_length: bool = True,
) -> List[Any]:
    """Generic wrapper to call a function on a list.

    This function let you crete custom functions in order to create columns.

    Args:
        list_of_values: A column (a list) or a list of lists (many columns).
        column_func: The function that you want to be used to tranform the
            `list_of_values`.
            Eg: "lambda (x): x"
        column_func_kwargs: Dictionary of the arguments that are going to be
            passed to the exploding fucntion.
            Eg: If the fucntion passed is `generate_dates`
            { "start_dt": "2019-01-01", "end_dt": "2020-01-01", "freq": "M",}.
        num_rows: Exact number of rows that will be created.
        resize: Resize is a boolean that will shorten or replicate the output
            column in case the len(output)!= len(table)
            Eg: True
        check_all_inputs_same_length: Boolean that explicit if the arrays from
            `list_of_values` have the same length.

    Returns:
        A list (column) of values with the desired function.

    """
    callable_to_apply = load_function_if_string(column_func)

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

    This function applies a function to a row. It will first join all the
    same sized columns from `list_of_values` into a new column.

    Args:
        list_of_values: A column (a list) or a list of lists (many columns).
        row_func: The function that you want to be used to tranform the
            `list_of_values`.
            Eg: "lambda (x): x"
        row_func_kwargs: Dictionary of the arguments that are going to be
            passed to the exploding fucntion.
            Eg: If the fucntion passed is `generate_dates`
            { "start_dt": "2019-01-01", "end_dt": "2020-01-01", "freq": "M",}.
        num_rows: Exact number of rows that will be created.
        resize: Resize is a boolean that will shorten or replicate the output
            column in case the len(output)!= len(table)
            Eg: True

    Returns:
        A list (column) of values with the desired function.
    """
    callable_to_apply = load_function_if_string(row_func)

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


def load_callable_with_libraries(function: str) -> Callable:
    """Evaluate a string function to convert into a callable.

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


def load_function_if_string(func):
    """Load function if it is string.

    No need for additional description.

    Args:
        func: Received function, can be a string or a function.
        Eg1: func = "lambda x: 'product'+str(x)"
        Eg2: Define function and pass it as a parameter:
            def _dummy(n):
            return [i + 1 for i in range(n)]
            func = _dummy
        Obs: When using Kedro Glass, it may not be recommended for
            defining new functions inside parameters.

    Returns:
        A function, whether if it is a lambda or a custom function.
    """
    if isinstance(func, str):
        if func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(func)
    else:
        callable_to_apply = func
    return callable_to_apply
