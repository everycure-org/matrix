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
"""This module provides tests for data fabrication utils"""

import json
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import yaml

from data_fabricator.v1.core.functions import (
    column_apply,
    conditional_generate_from_weights,
    cross_product,
    cross_product_with_separator,
    drop_duplicates,
    explode,
    faker,
    generate_dates,
    generate_random_arrays,
    generate_random_numbers,
    generate_single_date,
    generate_unique_id,
    generate_values,
    hash_string,
    load_callable_with_libraries,
    numpy_random,
    row_apply,
)

# pylint: skip-file
# flake8: noqa


def test_generate_unique_id():
    ids = generate_unique_id(
        num_rows=200,
    )

    assert len(ids) == 200


def test_generate_unique_id_prefix():
    ids = generate_unique_id(num_rows=400, prefix="id")

    has_prefix = [x.startswith("id") for x in ids]

    assert len(ids) == 400
    assert set(has_prefix) == {True}


def test_generate_unique_id_length():
    ids = generate_unique_id(num_rows=400, id_length=10)

    id_length = list(set([len(x) for x in ids]))

    assert id_length == [10]


def test_generate_unique_id_null():
    ids = generate_unique_id(num_rows=400, prefix="id", prob_null=0.5, seed=1)

    assert len(ids) == 400
    assert 0.45 <= ids.count(None) / len(ids) <= 0.55


def test_generate_unique_id_replace_null():
    ids = generate_unique_id(
        num_rows=400, prefix="id", prob_null=0.5, seed=1, null_value="unknown"
    )

    assert len(ids) == 400
    assert ids.count(None) == 0
    assert 0.45 <= ids.count("unknown") / len(ids) <= 0.55


def test_generate_unique_id_min():
    ids = generate_unique_id(
        num_rows=1000,
        id_start_range=5,
    )

    minimum_id = ids[0]

    assert minimum_id == "5"
    assert len(ids) == 1000


def test_generate_unique_id_down_sampled():
    ids = generate_unique_id(
        num_rows=100,
        id_start_range=1,
        id_end_range=1000,
    )

    ids_as_integers = [int(x) for x in ids]

    assert len(ids) == 100
    assert len(set(ids)) <= 1 + 1000
    assert len(set(ids)) == 100
    assert min(ids_as_integers) >= 1
    assert max(ids_as_integers) <= 1000


def test_generate_unique_id_up_sampled():
    ids = generate_unique_id(
        num_rows=10000,
        id_start_range=1,
        id_end_range=1000,
    )

    ids_as_integers = [int(x) for x in ids]

    assert len(ids) == 10000
    assert len(set(ids)) <= 1 + 1000
    assert min(ids_as_integers) >= 1
    assert max(ids_as_integers) <= 1000


def test_generate_random_numbers():
    random_numbers = generate_random_numbers(
        num_rows=500,
    )

    num_floats = len([x for x in random_numbers if isinstance(x, float)])

    assert len(random_numbers) == 500 == num_floats


def test_generate_random_arrays_integers():
    random_arrays = generate_random_arrays(num_rows=500, sample_values=[10, 100, 1000])

    assert len(random_arrays) == 500

    for sub_arr in random_arrays:
        assert set(sub_arr) - set([10, 100, 1000]) == set()


def test_generate_random_arrays_integers_seed():
    random_arrays = generate_random_arrays(
        num_rows=10, sample_values=[10, 100, 1000], seed=1
    )

    assert random_arrays == [
        [1000],
        [100],
        [100],
        [100, 1000],
        [10],
        [10, 100],
        [1000, 10],
        [100, 1000, 10],
        [10, 100, 1000],
        [10],
    ]


def test_generate_random_arrays_null():
    random_arrays = generate_random_arrays(
        num_rows=1000, sample_values=[10, 100, 1000], prob_null=0.2
    )

    assert len(random_arrays) == 1000
    assert 0.15 <= random_arrays.count(None) / len(random_arrays) <= 0.25


def test_generate_random_arrays_string():
    random_arrays = generate_random_arrays(
        num_rows=100,
        sample_values=["sample_1", "sample_2", "sample_3"],
    )

    assert len(random_arrays) == 100

    for sub_arr in random_arrays:
        assert set(sub_arr) - set(["sample_1", "sample_2", "sample_3"]) == set()


def test_generate_random_arrays_string_duplicates():
    random_arrays = generate_random_arrays(
        num_rows=10,
        sample_values=["sample_1", "sample_2", "sample_3"],
        allow_duplicates=True,
        seed=1,
    )

    assert len(random_arrays) == 10

    assert random_arrays == [
        ["sample_2"],
        ["sample_3"],
        ["sample_3", "sample_3"],
        ["sample_2", "sample_2"],
        ["sample_1", "sample_3"],
        ["sample_3", "sample_1"],
        ["sample_3", "sample_2", "sample_3"],
        ["sample_3"],
        ["sample_2"],
        ["sample_1"],
    ]


def test_generate_random_arrays_string_fixed_length():
    random_arrays = generate_random_arrays(
        num_rows=5, sample_values=["sample_1", "sample_2", "sample_3"], length=2, seed=1
    )

    assert len(random_arrays) == 5

    assert random_arrays == [
        ["sample_1", "sample_3"],
        ["sample_2", "sample_1"],
        ["sample_2", "sample_3"],
        ["sample_2", "sample_3"],
        ["sample_1", "sample_3"],
    ]


def test_generate_random_arrays_string_fixed_length_duplicates():
    random_arrays = generate_random_arrays(
        num_rows=5,
        sample_values=["sample_1", "sample_2", "sample_3"],
        length=2,
        allow_duplicates=True,
        seed=1,
    )

    assert len(random_arrays) == 5

    assert random_arrays == [
        ["sample_1", "sample_2"],
        ["sample_2", "sample_3"],
        ["sample_3", "sample_3"],
        ["sample_3", "sample_3"],
        ["sample_2", "sample_2"],
    ]


def test_generate_random_numbers_seed():
    random_numbers = generate_random_numbers(num_rows=10, end_range=10, seed=1)
    assert [round(num, 3) for num in random_numbers] == [
        1.344,
        8.474,
        7.638,
        2.551,
        4.954,
        4.495,
        6.516,
        7.887,
        0.939,
        0.283,
    ]


def test_generate_random_numbers_null():
    random_numbers = generate_random_numbers(num_rows=1000, prob_null=0.2)

    assert len(random_numbers) == 1000
    assert 0.15 <= random_numbers.count(None) / len(random_numbers) <= 0.25


def test_generate_random_numbers_ranges():
    random_numbers = generate_random_numbers(
        num_rows=100,
        start_range=5,
        end_range=105,
    )

    min_number = min(random_numbers)
    max_number = max(random_numbers)

    assert len(random_numbers) == 100
    assert 5 <= min_number
    assert max_number <= 105


def test_generate_values_from_list():
    input_values = ["a", "b", "c"]

    values = generate_values(num_rows=100, sample_values=input_values)

    assert len(values) == 100
    assert len(set(values)) == len(set(input_values))


def test_generate_values_from_numerical_list():
    input_values = [1, 2, 3]

    values = generate_values(num_rows=100, sample_values=input_values)

    assert len(values) == 100
    assert len(set(values)) == len(set(input_values))


def test_generate_values_from_dict():
    input_values = {"a": 100, "b": 10, "c": 1}

    values = generate_values(num_rows=100000, sample_values=input_values)

    sum_a = len([x for x in values if x == "a"])
    sum_b = len([x for x in values if x == "b"])
    sum_c = len([x for x in values if x == "c"])

    assert len(values) == 100000
    assert 9.5 <= (sum_a / sum_b) <= 10.5
    assert 9.5 <= (sum_b / sum_c) <= 10.5


def test_generate_values_from_list_null():
    input_values = ["a", "b", "c"]

    values = generate_values(num_rows=1000, sample_values=input_values, prob_null=0.7)

    non_none_values = [v for v in values if v]

    assert len(values) == 1000
    assert len(set(non_none_values)) == len(set(input_values))
    assert 0.65 <= values.count(None) / len(values) <= 0.75


def test_generate_values_exact():
    input_values = ["a", "b", "c", "d", "e"]

    values = generate_values(num_rows=5, sample_values=input_values)

    assert input_values == values


def test_generate_values_downsample():
    input_values = ["a", "b", "c", "d", "e"]

    values = generate_values(num_rows=4, sample_values=input_values)

    assert len(set(values)) == 4


def test_hash_string():
    input_list = [1, 2, 3, 1, 2, "a", "b", "b", "a", 3]

    element_instructions = {
        "row_func": hash_string,
        "row_func_kwargs": {
            "buckets": ["x", "y"],
        },
    }

    dependent_values = row_apply(list_of_values=input_list, **element_instructions)

    tuples = [(x, y) for x, y in zip(input_list, dependent_values)]

    unique_tuples = []
    for tpl in tuples:
        if tpl not in unique_tuples:
            unique_tuples.append(tpl)

    assert len(set(dependent_values)) == 2
    assert len(unique_tuples) == len(set(input_list))


def test_conditional_generate_from_weights():
    input_values = {"a": {"j": 10, "k": 1}, "b": {"m": 2, "n": 1, "p": 10}}

    values = conditional_generate_from_weights(
        value="b", dependent_weights=input_values
    )

    assert len(set(values)) == 1


def test_conditional_generate_from_weights_multiple_rows():
    input_values = {"a": {"j": 10, "k": 1}, "b": {"m": 2, "p": 1}}
    num_rows = 100000
    values = []
    for i in range(num_rows):
        value = conditional_generate_from_weights(
            value="b", dependent_weights=input_values
        )
        values.append(value)

    sum_m = len([x for x in values if x == "m"])
    sum_p = len([x for x in values if x == "p"])

    assert sum_m > sum_p
    assert 1.5 <= (sum_m / sum_p) <= 2.5


def test_row_apply_single_vector():
    categorical_values = ["a", "b", "c"]

    categorical_list = generate_values(
        num_rows=100000, sample_values=categorical_values, seed=1
    )

    element_instructions = {
        "row_func": conditional_generate_from_weights,
        "row_func_kwargs": {
            "dependent_weights": {
                "a": {
                    "j": 10,
                    "k": 1,
                },
                "b": {
                    "m": 2,
                    "n": 1,
                },
                "c": {
                    "y": 1,
                    "z": 1,
                },
            },
        },
    }

    dependent_values = row_apply(
        list_of_values=categorical_list, **element_instructions
    )

    sum_j = len([x for x in dependent_values if x == "j"])
    sum_k = len([x for x in dependent_values if x == "k"])

    sum_m = len([x for x in dependent_values if x == "m"])
    sum_n = len([x for x in dependent_values if x == "n"])

    sum_y = len([x for x in dependent_values if x == "y"])
    sum_z = len([x for x in dependent_values if x == "z"])

    assert len(dependent_values) == len(categorical_list)
    assert set(dependent_values) == {"j", "k", "m", "n", "y", "z"}
    assert 9.5 <= (sum_j / sum_k) <= 10.5
    assert 1.5 <= (sum_m / sum_n) <= 2.5
    assert 0.5 <= (sum_y / sum_z) <= 1.5


def test_row_apply_multiple_vector():
    ids1 = generate_unique_id(
        num_rows=200,
    )
    ids2 = generate_unique_id(
        num_rows=200,
    )
    ids3 = generate_unique_id(
        num_rows=200,
    )

    new_values = row_apply(
        list_of_values=[ids1, ids2, ids3],
        row_func=lambda x, y, z: int(x) + int(y) + int(z),
    )

    assert len(new_values) == 200
    assert min(new_values) == 3
    assert max(new_values) == 600


def test_row_apply_lambda():
    ids1 = generate_unique_id(
        num_rows=200,
    )
    ids2 = generate_unique_id(
        num_rows=200,
    )
    ids3 = generate_unique_id(
        num_rows=200,
    )

    new_values = row_apply(
        list_of_values=[ids1, ids2, ids3],
        row_func="lambda x, y, z: int(x) + int(y) + int(z)",
    )

    assert len(new_values) == 200
    assert min(new_values) == 3
    assert max(new_values) == 600


def test_drop_duplicates_single():
    ids1 = generate_unique_id(num_rows=1000, id_start_range=1, id_end_range=11)

    no_dupes = drop_duplicates(ids1)
    assert len(no_dupes) == 10
    assert int(max(no_dupes)) == 9
    assert int(min(no_dupes)) == 1


def test_drop_duplicates_multiple():
    ids1 = [1, 2, 3, 3, 4, 4, 5]
    ids2 = [1, 2, 3, 3, 4, 5, 5]

    no_dupes1 = drop_duplicates(ids1, ids2, position=0)
    no_dupes2 = drop_duplicates(ids1, ids2, position=1)

    assert no_dupes1 == [1, 2, 3, 4, 4, 5]
    assert no_dupes2 == [1, 2, 3, 4, 5, 5]


def test_column_apply_single():
    ids = generate_unique_id(
        num_rows=200,
    )
    new_values = column_apply(
        list_of_values=ids, column_func=lambda x: [v for v in x if int(v) > 100]
    )
    assert len(new_values) == 100


def test_column_apply_multiple():
    ids1 = generate_unique_id(
        num_rows=200,
    )
    ids2 = generate_unique_id(
        num_rows=200,
    )
    new_values = column_apply(
        list_of_values=[ids1, ids2],
        column_func=lambda x, y: [(v1, v2) for v1, v2 in zip(x, y)],
    )
    assert len(new_values) == 200


def test_column_apply_null():
    ids = generate_unique_id(num_rows=1000)
    new_values = column_apply(
        list_of_values=ids,
        column_func=lambda x: [v for v in x if int(v) > 100],
        prob_null=0.1,
    )
    assert len(new_values) == 900
    assert 0.05 <= new_values.count(None) / len(new_values) <= 0.15


def test_generate_random_dates():
    dates = generate_dates(
        num_rows=200, start_dt="2019-01-01", end_dt="2020-12-31", freq="M"
    )

    assert len(dates) == 200
    assert str(min(dates).date()) == "2019-01-31"
    assert str(max(dates).date()) == "2020-12-31"


def test_generate_sorted_random_dates():
    dates = generate_dates(
        num_rows=200,
        start_dt="2019-01-01",
        end_dt="2020-12-31",
        freq="M",
        sort_dates=True,
    )

    assert len(dates) == 200
    assert str(min(dates).date()) == "2019-01-31"
    assert str(max(dates).date()) == "2020-12-31"
    assert all(x <= y for x, y in zip(dates, dates[1:]))


def test_generate_single_date():
    single_date = generate_single_date(start_dt="2019-01-01", end_dt="2020-12-31")

    assert len(single_date) == 1
    assert single_date[0] > "2019-01-01"
    assert single_date[0] < "2020-12-31"


@pytest.mark.parametrize(
    "date_format,expected_boundaries",
    [
        ("%m/%d/%Y %I:%M:%S %p", ["01/31/2019 12:00:00 AM", "12/31/2020 12:00:00 AM"]),
        ("%m/%d/%Y", ["01/31/2019", "12/31/2020"]),
    ],
)
def test_generate_dates_with_format(date_format, expected_boundaries):
    dates = generate_dates(
        num_rows=200,
        start_dt="2019-01-01",
        end_dt="2020-12-31",
        freq="M",
        date_format=date_format,
        sort_dates=True,
    )

    assert len(dates) == 200
    assert dates[0] == expected_boundaries[0]
    assert dates[-1] == expected_boundaries[1]


def test_numpy_random1():
    ints = numpy_random(num_rows=10, distribution="binomial", n=1, p=0.5, numpy_seed=1)
    assert ints == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]


def test_numpy_random2():
    ints = numpy_random(num_rows=10, distribution="pareto", a=1, numpy_seed=1)

    rounded = [round(x, 2) for x in ints]
    assert rounded == [0.72, 2.58, 0.0, 0.43, 0.17, 0.1, 0.23, 0.53, 0.66, 1.17]


def test_explode_single_vector():
    l1 = [[1, 2, 3]]

    def _dummy(n):
        return [i + 1 for i in range(n)]

    result0 = explode(
        list_of_values=l1, explode_func=_dummy, explode_func_kwargs={"n": 3}, position=0
    )

    result1 = explode(
        list_of_values=l1, explode_func=_dummy, explode_func_kwargs={"n": 3}, position=1
    )

    assert result0 == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result1 == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_explode_multiple_vectors():
    l1 = [[1, 2, 3], ["a", "b", "c"]]

    def _dummy(n):
        return [i + 1 for i in range(n)]

    result0 = explode(
        list_of_values=l1, explode_func=_dummy, explode_func_kwargs={"n": 3}, position=0
    )

    result1 = explode(
        list_of_values=l1, explode_func=_dummy, explode_func_kwargs={"n": 3}, position=1
    )

    result2 = explode(
        list_of_values=l1, explode_func=_dummy, explode_func_kwargs={"n": 3}, position=2
    )

    assert result0 == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result1 == ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    assert result2 == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_explode_with_distribution():
    list_of_values = [[1, 2, 3], [1, 2, 3]]

    results = explode(
        list_of_values=list_of_values,
        distribution_kwargs={
            "distribution": "gamma",
            "scale": 10,
            "shape": 1,
            "numpy_seed": 1,
        },
        explode_func=generate_dates,
        explode_func_kwargs={
            "start_dt": "2019-01-01",
            "end_dt": "2020-01-01",
            "freq": "M",
        },
        position=0,
    )

    num_ones = len([x for x in results if x == 1])
    num_twos = len([x for x in results if x == 2])
    num_threes = len([x for x in results if x == 3])
    assert num_ones == 5
    assert num_twos == 12
    assert num_threes == 12


def test_faker1():
    result = faker(
        num_rows=10,
        provider="company",
    )

    assert len(result) == 10


def test_faker2():
    result = faker(
        num_rows=10,
        provider="name",
        faker_seed=1,
    )
    assert len(result) == 10
    assert result == [
        "Ryan Gallagher",
        "Jon Cole",
        "Rachel Davis",
        "Russell Reynolds",
        "April Griffin",
        "Crystal Landry",
        "Amanda Johnson",
        "Teresa James",
        "Javier Johnson",
        "Jeffrey Simpson",
    ]


def test_faker_nulls():
    result = faker(num_rows=1000, provider="name", seed=1, prob_null=0.5)

    assert len(result) == 1000
    assert 0.45 <= result.count(None) / len(result) <= 0.55


def test_load_callable_with_libraries():
    functions_to_test = [
        "lambda x: x - datetime.timedelta(weeks=2*random.random())",
        "lambda x: int(x)",
        "lambda x: x + random.random()",
        "lambda x: max(x * (0.5 + random.random()), 0)",
        "lambda x: x.day()",
        "lambda x: calendar.monthrange(year=2020, month=x)",
    ]

    for function in functions_to_test:
        assert isinstance(load_callable_with_libraries(function), Callable)

    # Test if imports are available globally.
    load_callable_with_libraries(functions_to_test[5])(1)


def test_cross_product():
    l1 = ["p1", "p2", "p3"]
    l2 = ["q1", "q2"]
    l3 = ["r1", "r2"]
    values = cross_product(l1, l2, l3, position=1)

    assert len(set(values)) == 2
    assert values == [
        "q1",
        "q1",
        "q2",
        "q2",
        "q1",
        "q1",
        "q2",
        "q2",
        "q1",
        "q1",
        "q2",
        "q2",
    ]


def test_cross_product_with_separator():
    l1 = ["car", "bike", "bus"]
    l2 = ["hybrid", "plugin"]
    values = cross_product_with_separator(l1, l2, separator=" ")

    assert values == [
        "car hybrid",
        "car plugin",
        "bike hybrid",
        "bike plugin",
        "bus hybrid",
        "bus plugin",
    ]
