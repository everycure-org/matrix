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

from data_fabricator.v0.core.fabricator import (
    MockDataGenerator,
    column_apply,
    conditional_generate_from_weights,
    cross_product,
    drop_duplicates,
    drop_filtered_condition_rows,
    explode,
    faker,
    generate_dates,
    generate_random_arrays,
    generate_random_numbers,
    generate_single_date,
    generate_unique_id,
    generate_values,
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


def test_generate_random_numbers():
    random_numbers = generate_random_numbers(
        num_rows=500,
    )

    num_floats = len([x for x in random_numbers if isinstance(x, float)])

    assert len(random_numbers) == 500 == num_floats


def test_generate_random_numbers_integers():
    random_numbers = generate_random_numbers(
        num_rows=1000,
        end_range=1000,
        integer=True,
    )

    num_ints = len([x for x in random_numbers if isinstance(x, int)])

    assert len(random_numbers) == 1000 == num_ints


def test_generate_random_numbers_integers_seed():
    random_numbers = generate_random_numbers(
        num_rows=10, end_range=10, integer=True, seed=1
    )
    assert random_numbers == [1, 8, 7, 2, 4, 4, 6, 7, 0, 0]


def test_generate_random_numbers_null():
    random_numbers = generate_random_numbers(num_rows=1000, prob_null=0.2)

    assert len(random_numbers) == 1000
    assert 0.15 <= random_numbers.count(None) / len(random_numbers) <= 0.25


def test_generate_random_numbers_ranges():
    random_numbers = generate_random_numbers(
        num_rows=100,
        start_range=5,
        end_range=105,
        integer=True,
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
        "row_func": "hash_string",
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


def test_drop_filtered_condition_rows():
    l1 = [1, 2, 3, 4]
    l2 = ["a", "b", "c", "d"]
    l3 = [True, True, False, True]

    values = drop_filtered_condition_rows(l1, l2, l3, position=1)

    assert len(set(values)) == 3
    assert values == ["a", "b", "d"]


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
        "row_func": "conditional_generate_from_weights",
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


def test_global_class():
    string = """
    customers:
        num_rows: 200
        columns:
            hcp_id:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            hcp_name:
                type: faker
                provider: name
            hcp_company:
                type: faker
                provider: company
            hcp_phone_number:
                type: faker
                provider: phone_number
                localisation: ja-JP
            ftr1:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
            ftr2:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: True
            ftr3:
                type: generate_random_numbers
                start_range: 200
                end_range: 1000
                integer: True
            ftr4:
                type: generate_random_numbers

    stage3_sales:
        columns:
            hcp_id:
                type: explode
                list_of_values: customers.hcp_id
                explode_func: generate_dates
                explode_func_kwargs:
                    start_dt: 2019-01-31
                    end_dt: 2020-12-31
                    freq: M
                position: 0
            date_month:
                type: explode
                list_of_values: customers.hcp_id
                explode_func: generate_dates
                explode_func_kwargs:
                    start_dt: 2019-01-31
                    end_dt: 2020-12-31
                    freq: M
                position: 1
            interaction:
                type: generate_values
                sample_values:
                    interaction1: 80
                    interaction2: 20
                    interaction-other: 1
            product_cd:
                type: generate_values
                sample_values: ["R03AC02","R03AC03","R03AC04","R03AC08","R03AC17"]
            channel_cd:
                type: row_apply
                list_of_values: stage3_sales.product_cd
                row_func: conditional_generate_from_weights
                row_func_kwargs:
                    dependent_weights:
                        R03AC02:
                            a: 1
                            b: 1
                        R03AC03:
                            c: 4
                            d: 1
                        R03AC04:
                            e: 10
                            f: 1
                        R03AC08:
                            g: 1
                        R03AC17:
                            x: 2
                            y: 1
            sales:
                type: row_apply
                list_of_values: [customers.ftr3, customers.ftr4]
                row_func: "lambda x,y: 2 * int(x) + 100 * int(y) + 100 * random.random()"
                resize: True
            sales_lambda:
                type: row_apply
                list_of_values: stage3_sales.sales
                row_func: "lambda x: x+10"

    channel_concepts:
        columns:
            channel_cd:
                type: column_apply
                list_of_values: [stage3_sales.channel_cd, stage3_sales.product_cd]
                column_func: drop_duplicates
                column_func_kwargs:
                    position: 0
            product_cd:
                type: column_apply
                list_of_values: [stage3_sales.channel_cd, stage3_sales.product_cd]
                column_func: drop_duplicates
                column_func_kwargs:
                    position: 1
            channel_description:
                type: row_apply
                list_of_values: channel_concepts.channel_cd
                row_func: conditional_string
                row_func_kwargs:
                    mapping:
                        a: alpha
                        b: bravo
                        c: charlie
                        d: delta
                        e: echo
                        f: foxtrot
                        g: golf
                        x: x-ray
                        y: yankee
    """

    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    mock_generator.all_dataframes["customers"].head(10)
    mock_generator.all_dataframes["stage3_sales"].head(10)
    mock_generator.all_dataframes["channel_concepts"].head(10)

    assert mock_generator.all_dataframes["customers"].shape == (200, 8)
    assert mock_generator.all_dataframes["stage3_sales"].shape == (4800, 7)
    assert (
        int(
            (
                mock_generator.all_dataframes["stage3_sales"]["sales_lambda"]
                - mock_generator.all_dataframes["stage3_sales"]["sales"]
            ).mean()
        )
        == 10
    )
    assert mock_generator.all_dataframes["customers"]["hcp_id"].nunique() == 200
    assert mock_generator.all_dataframes["customers"]["hcp_id"].nunique() <= 200
    assert mock_generator.all_dataframes[
        "channel_concepts"
    ].drop_duplicates().shape == (9, 3)


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


def test_lambda_tuple_ordering():
    string = """
    internal_external_hcp_mapping:
        num_rows: 90
        columns:
            provider_id:
                type: generate_unique_id
                prefix: provider_
                id_start_range: 10
                id_end_range: 100
                id_length: 12
            customer_id:
                type: generate_unique_id
                prefix: customer_
                id_start_range: 0
                id_end_range: 500
                id_length: 12
                prob_null: 0.5
                seed: 1
            provider_type:
                type: generate_values
                sample_values: ["gp", "specialist", "pharmacists"]
                seed: 1

    _TEMP_all_events:
        num_rows: 10000
        columns:
            provider_id:
                type: row_apply
                list_of_values: [
                    internal_external_hcp_mapping.provider_id,
                    internal_external_hcp_mapping.customer_id,
                    internal_external_hcp_mapping.provider_type
                ]
                row_func: "lambda x, y, z: x"
                resize: True
                seed: 1
            customer_id:
                type: row_apply
                list_of_values: [
                    internal_external_hcp_mapping.provider_id,
                    internal_external_hcp_mapping.customer_id,
                    internal_external_hcp_mapping.provider_type
                ]
                row_func: "lambda x, y, z: y"
                resize: True
                seed: 1
            provider_type:
                type: row_apply
                list_of_values: [
                    internal_external_hcp_mapping.provider_id,
                    internal_external_hcp_mapping.customer_id,
                    internal_external_hcp_mapping.provider_type
                ]
                row_func: "lambda x, y, z: z"
                resize: True
                seed: 1
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    original = (
        mock_generator.all_dataframes["internal_external_hcp_mapping"]
        .sort_values(["provider_id", "customer_id", "provider_type"])
        .reset_index(drop=True)
    )
    deduped = (
        mock_generator.all_dataframes["_TEMP_all_events"]
        .drop_duplicates()
        .sort_values(["provider_id", "customer_id", "provider_type"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(original, deduped)


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


def test_parse_column_reference_for_generate_values():
    string = """
    products:
      num_rows: 2
      columns:
        id:
          type: generate_unique_id
          id_start_range: 0
          id_end_range: 2
          id_length: 4

    interactions:
      num_rows: 120
      columns:
        id:
          type: generate_unique_id
          id_start_range: 0
          id_end_range: 120
          id_length: 4
        product_cd:
          type: generate_values
          sample_values: products.id
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    assert len(mock_generator.all_dataframes["interactions"].columns) == 2
    assert (
        mock_generator.all_dataframes["interactions"].shape[0]
        == mock_generator.all_dataframes["interactions"]["id"].nunique()
    )
    assert (
        mock_generator.all_dataframes["interactions"]["product_cd"].nunique()
        <= mock_generator.all_dataframes["products"]["id"].nunique()
    )


def test_parse_column_reference_for_single_dt():
    string = """
     customers:
       num_rows: 10
       columns:
         id:
           type: generate_unique_id
           id_start_range: 0
           id_end_range: 10
           id_length: 4
         acct_enroll_dt:
           type: generate_dates
           start_dt: 2019-01-01
           end_dt: 2020-12-31
           freq: B
           seed: 1

     interactions:
       num_rows: 10
       columns:
         id:
           type: generate_unique_id
           id_start_range: 0
           id_end_range: 10
           id_length: 2
         customer_id:
           type: row_apply
           list_of_values:
             - customers.id
             - customers.acct_enroll_dt
           row_func: "lambda x, y: f'{x}'"
           resize: True
           seed: 1
         interaction_dt:
           type: row_apply
           list_of_values:
             - customers.id
             - customers.acct_enroll_dt
           row_func: "lambda x, y: generate_single_date(start_dt= y, end_dt= datetime.datetime.strptime(str(y.date()), '%Y-%m-%d') + datetime.timedelta(days=14))[0]"
           resize: True
           seed: 1
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    customers = mock_generator.all_dataframes["customers"]
    interactions = mock_generator.all_dataframes["interactions"]

    assert len(interactions.columns) == 3
    assert interactions.shape[0] == interactions["id"].nunique()
    assert interactions["interaction_dt"].nunique() <= 10
    assert interactions["customer_id"].nunique() == customers["id"].nunique()

    interactions["interaction_dt"] = pd.to_datetime(
        interactions["interaction_dt"], format="%Y-%m-%d"
    )
    for acct_id in interactions["customer_id"]:
        date_diff = abs(
            customers[customers["id"] == acct_id]["acct_enroll_dt"]
            - interactions[interactions["customer_id"] == acct_id]["interaction_dt"]
        )
        assert list(date_diff.dt.days) <= [14]


def test_parse_column_reference_for_drop_filtered_condition_rows():
    string = """
     account:
       num_rows: 10
       columns:
         acct_id:
           type: generate_unique_id
           id_start_range: 0
           id_end_range: 10
           id_length: 4
         acct_type:
           type: generate_values
           sample_values:
            - savings
            - current
         acct_start_dt:
           type: generate_dates
           start_dt: 2020-01-01
           end_dt: 2020-12-31
           freq: M
           seed: 1

     _temp_savings_account:
       num_rows: 10
       columns:
         acct_id:
           type: row_apply
           list_of_values: [account.acct_id, account.acct_start_dt]
           row_func: "lambda x,y: x"
         acct_start_dt:
           type: row_apply
           list_of_values: [account.acct_id, account.acct_start_dt]
           row_func: "lambda x,y: y"
         principal_amount:
           type: generate_random_numbers
           start_range: 30000
           end_range: 200000
         keep_row:
           type: row_apply
           list_of_values: account.acct_type
           row_func: "lambda x: x=='savings'"

     savings_account:
      columns:
         acct_id:
           type: column_apply
           list_of_values:
            - _temp_savings_account.acct_id
            - _temp_savings_account.principal_amount
            - _temp_savings_account.keep_row
            - _temp_savings_account.acct_start_dt
           column_func: drop_filtered_condition_rows
           column_func_kwargs:
              position: 0
         principal_amount:
           type: column_apply
           list_of_values:
            - _temp_savings_account.acct_id
            - _temp_savings_account.principal_amount
            - _temp_savings_account.keep_row
            - _temp_savings_account.acct_start_dt
           column_func: drop_filtered_condition_rows
           column_func_kwargs:
              position: 1
         acct_start_dt:
           type: column_apply
           list_of_values:
            - _temp_savings_account.acct_id
            - _temp_savings_account.principal_amount
            - _temp_savings_account.keep_row
            - _temp_savings_account.acct_start_dt
           column_func: drop_filtered_condition_rows
           column_func_kwargs:
              position: 3

    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    account = mock_generator.all_dataframes["account"]
    savings_account = mock_generator.all_dataframes["savings_account"]
    temp_savings_account = mock_generator.all_dataframes["_temp_savings_account"]
    saving_account_ids = account[account.acct_type == "savings"]["acct_id"]

    assert len(account.columns) == 3
    assert len(savings_account.columns) == 3
    assert savings_account["acct_id"].nunique() == len(saving_account_ids)
    for id in saving_account_ids:
        assert list(
            temp_savings_account[temp_savings_account["acct_id"] == id][
                "principal_amount"
            ]
        ) == list(savings_account[savings_account["acct_id"] == id]["principal_amount"])


def test_parse_column_reference_for_explode():
    string = """
    customers:
      num_rows: 2
      columns:
        id:
          type: generate_unique_id
          id_start_range: 0
          id_end_range: 2
          id_length: 4

    interactions:
      num_rows: 10
      columns:
        id:
          type: generate_unique_id
          id_start_range: 0
          id_end_range: 10
          id_length: 2
        customer_id:
          type: explode
          list_of_values: customers.id
          explode_func: generate_dates
          explode_func_kwargs:
            start_dt: "2019-01-01"
            end_dt: "2020-01-01"
            freq: B
            num_rows: 5
          position: 0
        interaction_dt:
          type: explode
          list_of_values: customers.id
          explode_func: generate_dates
          explode_func_kwargs:
            start_dt: "2009-01-01"
            end_dt: "2010-01-01"
            freq: B
            num_rows: 5
          position: 1
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    assert len(mock_generator.all_dataframes["interactions"].columns) == 3
    assert (
        mock_generator.all_dataframes["interactions"].shape[0]
        == mock_generator.all_dataframes["interactions"]["id"].nunique()
    )
    assert (
        mock_generator.all_dataframes["interactions"]["interaction_dt"].nunique() >= 5
    )
    assert (
        mock_generator.all_dataframes["interactions"]["interaction_dt"].nunique() <= 10
    )
    assert (
        mock_generator.all_dataframes["interactions"]["customer_id"].nunique()
        == mock_generator.all_dataframes["customers"]["id"].nunique()
    )


def test_parse_actual_column_reference_for_explode():
    string = """
     accounts:
       num_rows: 10
       columns:
         id:
           type: generate_unique_id
           id_start_range: 0
           id_end_range: 10
           id_length: 4
         acct_start_dt:
           type: generate_dates
           start_dt: 2020-01-01
           end_dt: 2020-12-31
           freq: M
           seed: 1

     transaction_statements:
       num_rows: 20
       columns:
         transaction_id:
           type: generate_unique_id
           id_start_range: 00
           id_end_range: 20
           id_length: 6
         acct_id:
           type: explode
           list_of_values:
             - accounts.id
             - accounts.acct_start_dt
           explode_func: generate_dates
           explode_func_kwargs:
            start_dt: list_of_values[1]
            end_dt: 2020-12-31
            freq: M
            num_rows: 2
           position: 0
         transaction_dt:
           type: explode
           list_of_values:
             - accounts.id
             - accounts.acct_start_dt
           explode_func: generate_dates
           explode_func_kwargs:
            start_dt: list_of_values[1]
            end_dt: 2020-12-31
            freq: M
            num_rows: 2
           position: 2
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    transaction_statements = mock_generator.all_dataframes["transaction_statements"]
    accounts = mock_generator.all_dataframes["accounts"]

    assert len(transaction_statements.columns) == 3
    assert (
        transaction_statements.shape[0]
        == transaction_statements["transaction_id"].nunique()
    )
    assert transaction_statements["acct_id"].nunique() == accounts["id"].nunique()

    for index, row in accounts.iterrows():
        assert list(accounts[accounts["id"] == row["id"]]["acct_start_dt"]) < list(
            transaction_statements[transaction_statements["acct_id"] == row["id"]][
                "transaction_dt"
            ]
        )


def test_parse_multiple_column_references_for_explode():
    string = """
    car_shapes:
      num_rows: 3
      columns:
        shape:
          type: generate_values
          sample_values:
            - sedan
            - coupe
            - hatchback
    car_engines:
      num_rows: 3
      columns:
        engine:
          type: generate_values
          sample_values:
            - electric
            - gasoline
            - diesel
    car_extras:
      num_rows: 9
      columns:
        shape:
          type: explode
          list_of_values:
            - car_shapes.shape
            - car_engines.engine
          explode_func: generate_unique_id
          explode_func_kwargs:
            id_start_range: 0
            id_end_range: 5
            id_length: 2
            num_rows: 3
          position: 0
        engine:
          type: explode
          list_of_values:
            - car_shapes.shape
            - car_engines.engine
          explode_func: generate_unique_id
          explode_func_kwargs:
            id_start_range: 0
            id_end_range: 5
            id_length: 2
            num_rows: 3
          position: 1
        extra_cd:
          type: explode
          list_of_values:
            - car_shapes.shape
            - car_engines.engine
          explode_func: generate_unique_id
          explode_func_kwargs:
            id_start_range: 0
            id_end_range: 5
            id_length: 2
            num_rows: 3
          position: 2
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    assert len(mock_generator.all_dataframes["car_extras"].columns) == 3
    assert (
        mock_generator.all_dataframes["car_extras"]["shape"].nunique()
        == mock_generator.all_dataframes["car_shapes"]["shape"].nunique()
    )
    assert (
        mock_generator.all_dataframes["car_extras"]["engine"].nunique()
        == mock_generator.all_dataframes["car_engines"]["engine"].nunique()
    )
    assert (
        mock_generator.all_dataframes["car_extras"]
        .groupby(["shape", "engine", "extra_cd"])
        .ngroups
        == mock_generator.all_dataframes["car_extras"].shape[0]
    )


def test_parse_column_reference_for_explode_to_generate_values():
    string = """
    car_shapes:
      num_rows: 4
      columns:
        shape:
          type: generate_values
          sample_values:
            - sedan
            - coupe
            - hatchback
            - truck
    car_cylinders:
      num_rows: 4
      columns:
        cylinder:
          type: generate_values
          sample_values: [3, 4, 6, 8]
    car_colours:
      num_rows: 4
      columns:
        colour:
          type: generate_values
          sample_values:
            - black
            - red
            - green
            - purple
    colour_combinations:
      num_rows: 12
      columns:
        shape:
          type: explode
          list_of_values: car_shapes.shape
          explode_func: generate_values
          explode_func_kwargs:
            sample_values: car_colours.colour
            num_rows: 3
          position: 0
        colour:
          type: explode
          list_of_values: car_shapes.shape
          explode_func: generate_values
          explode_func_kwargs:
            sample_values: car_colours.colour
            num_rows: 3
          position: 1
    cylinder_combinations:
      num_rows: 8
      columns:
        shape:
          type: explode
          list_of_values: car_shapes.shape
          explode_func: generate_values
          explode_func_kwargs:
            sample_values: car_cylinders.cylinder
            num_rows: 2
          position: 0
        cylinder:
          type: explode
          list_of_values: car_shapes.shape
          explode_func: generate_values
          explode_func_kwargs:
            sample_values: car_cylinders.cylinder
            num_rows: 2
          position: 1
    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    assert len(mock_generator.all_dataframes["colour_combinations"].columns) == 2
    assert (
        mock_generator.all_dataframes["colour_combinations"]["shape"].nunique()
        == mock_generator.all_dataframes["car_shapes"]["shape"].nunique()
    )
    assert mock_generator.all_dataframes["colour_combinations"]["colour"].nunique() >= 3

    assert len(mock_generator.all_dataframes["cylinder_combinations"].columns) == 2
    assert (
        mock_generator.all_dataframes["cylinder_combinations"]["shape"].nunique()
        == mock_generator.all_dataframes["car_shapes"]["shape"].nunique()
    )
    assert (
        mock_generator.all_dataframes["cylinder_combinations"]["cylinder"].nunique()
        >= 2
    )


def test_parse_column_reference_for_cross_product():
    string = """
            customers:
              num_rows: 2
              columns:
                customer_id:
                  type: generate_unique_id
                  prefix: customer_

            products:
              num_rows: 2
              columns:
                product_id:
                  type: generate_unique_id
                  prefix: product_

            dates:
              num_rows: 3
              columns:
                date:
                  type: generate_dates
                  start_dt: 2020-01-01
                  end_dt: 2020-01-03
                  freq: D
            cross_product:
              columns:
                customer_id:
                  type: column_apply
                  check_all_inputs_same_length: False
                  list_of_values:
                    - customers.customer_id
                    - products.product_id
                    - dates.date
                  column_func: cross_product
                  column_func_kwargs:
                    position: 0
                product_id:
                  type: column_apply
                  check_all_inputs_same_length: False
                  resize: True
                  list_of_values:
                    - customers.customer_id
                    - products.product_id
                    - dates.date
                  column_func: cross_product
                  column_func_kwargs:
                    position: 1
                date:
                  type: column_apply
                  check_all_inputs_same_length: False
                  list_of_values:
                    - customers.customer_id
                    - products.product_id
                    - dates.date
                  column_func: cross_product
                  column_func_kwargs:
                    position: 2

    """
    parsed_dict = yaml.safe_load(string)

    mock_generator = MockDataGenerator(instructions=parsed_dict)

    mock_generator.generate_all()

    customers = mock_generator.all_dataframes["customers"]
    products = mock_generator.all_dataframes["products"]
    dates = mock_generator.all_dataframes["dates"]
    cross_product = mock_generator.all_dataframes["cross_product"]
    assert len(customers) == 2
    assert len(products) == 2
    assert len(dates) == 3
    assert len(cross_product) == 12

    data = json.loads(cross_product.to_json(orient="records", date_format="iso"))

    # pandas<1.5.0 creates a zero-time timestamp, whereas pandas>=1.5.0 creates
    # a UTC timestamp - make sure both will work
    data = [{k: v.rstrip("Z") for k, v in _data.items()} for _data in data]

    assert data == [
        {
            "customer_id": "customer_1",
            "product_id": "product_1",
            "date": "2020-01-01T00:00:00.000",
        },
        {
            "customer_id": "customer_1",
            "product_id": "product_1",
            "date": "2020-01-02T00:00:00.000",
        },
        {
            "customer_id": "customer_1",
            "product_id": "product_1",
            "date": "2020-01-03T00:00:00.000",
        },
        {
            "customer_id": "customer_1",
            "product_id": "product_2",
            "date": "2020-01-01T00:00:00.000",
        },
        {
            "customer_id": "customer_1",
            "product_id": "product_2",
            "date": "2020-01-02T00:00:00.000",
        },
        {
            "customer_id": "customer_1",
            "product_id": "product_2",
            "date": "2020-01-03T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_1",
            "date": "2020-01-01T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_1",
            "date": "2020-01-02T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_1",
            "date": "2020-01-03T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_2",
            "date": "2020-01-01T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_2",
            "date": "2020-01-02T00:00:00.000",
        },
        {
            "customer_id": "customer_2",
            "product_id": "product_2",
            "date": "2020-01-03T00:00:00.000",
        },
    ]


def test_parse_column_reference_for_cross_product_position_value_error():
    string_err = """
                customers:
                  num_rows: 2
                  columns:
                    customer_id:
                      type: generate_unique_id
                      prefix: customer_

                products:
                  num_rows: 2
                  columns:
                    product_id:
                      type: generate_unique_id
                      prefix: product_

                dates:
                  num_rows: 2
                  columns:
                    date:
                      type: generate_dates
                      start_dt: 2020-01-01
                      end_dt: 2020-01-02
                      freq: D
                cross_product:
                  columns:
                    customer_id:
                      type: column_apply
                      list_of_values:
                        - customers.customer_id
                        - products.product_id
                        - dates.date
                      column_func: cross_product
                      column_func_kwargs:
                        position: 0
                    product_id:
                      type: column_apply
                      resize: True
                      list_of_values:
                        - customers.customer_id
                        - products.product_id
                        - dates.date
                      column_func: cross_product
                      column_func_kwargs:
                        position: 1
                    date:
                      type: column_apply
                      list_of_values:
                        - customers.customer_id
                        - products.product_id
                        - dates.date
                      column_func: cross_product
                      column_func_kwargs:
                        position: 3

        """
    parsed_dict_err = yaml.safe_load(string_err)

    mock_generator_err = MockDataGenerator(instructions=parsed_dict_err)

    error_string = "Position out of range. Should vary between 0 and 2, but received 3."

    with pytest.raises(ValueError, match=error_string):
        mock_generator_err.generate_all()


def test_mock_generator_seed():
    string = """
     accounts:
       num_rows: 10
       columns:
         id:
           type: generate_unique_id
           id_start_range: 0
           id_end_range: 10
           id_length: 4
         random:
           type: numpy_random
           distribution: binomial
           n: 1
           p: 0.5
         member_id:
           type: generate_unique_id
           prefix: mem_
           id_start_range: 0
           id_end_range: 5000
           id_length: 10
         period_start_date:
           type: generate_dates
           start_dt: 2019-01-01
           end_dt: 2021-01-01
           freq: D
         period_end_date:
           type: row_apply
           list_of_values: accounts.period_start_date
           row_func: "lambda x: x + datetime.timedelta(days=random.randint(100, 365))"
    """

    parsed_dict1 = yaml.safe_load(string)
    parsed_dict2 = yaml.safe_load(string)
    parsed_dict3 = yaml.safe_load(string)

    mock_generator1 = MockDataGenerator(seed=1, instructions=parsed_dict1)
    mock_generator1.generate_all()
    result1 = mock_generator1.all_dataframes["accounts"]

    mock_generator2 = MockDataGenerator(seed=1, instructions=parsed_dict2)
    mock_generator2.generate_all()
    result2 = mock_generator2.all_dataframes["accounts"]

    mock_generator3 = MockDataGenerator(instructions=parsed_dict3)
    mock_generator3.generate_all()
    result3 = mock_generator3.all_dataframes["accounts"]

    pd.testing.assert_frame_equal(result1, result2)

    with pytest.raises(AssertionError):
        # expect this to fail because seed not set
        pd.testing.assert_frame_equal(result1, result3)


def test_natural_columns_dtypes():
    string = """
    customers:
        num_rows: 50
        columns:
            id_string:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            grn_01_float:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
            grn_02_integer:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: True
            grn_03_integer:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: true
                prob_null: 0.5
            grn_04_float:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: False
                prob_null: 0.5
            gdt_05_date:
              type: generate_dates
              start_dt: 2019-01-01
              end_dt: 2020-12-31
              freq: M
            gvl_01_bool:
              type: generate_values 
              sample_values:
                [True, False]
            gvl_02_object:
              type: generate_values 
              sample_values:
                [True, False, None]
    """

    parsed_dict = yaml.safe_load(string)
    mock_generator = MockDataGenerator(instructions=parsed_dict)
    mock_generator.generate_all()
    output_dicts = mock_generator.all_dataframes
    assert isinstance(output_dicts["customers"]["id_string"].dtype, pd.StringDtype)
    assert isinstance(output_dicts["customers"]["grn_01_float"].dtype, pd.Float64Dtype)
    assert isinstance(output_dicts["customers"]["grn_02_integer"].dtype, pd.Int64Dtype)
    assert isinstance(output_dicts["customers"]["grn_03_integer"].dtype, pd.Int64Dtype)
    assert isinstance(output_dicts["customers"]["grn_04_float"].dtype, pd.Float64Dtype)
    assert output_dicts["customers"]["gdt_05_date"].dtype == np.dtype("<M8[ns]")
    assert isinstance(output_dicts["customers"]["gvl_01_bool"].dtype, pd.BooleanDtype)
    assert isinstance(output_dicts["customers"]["gvl_02_object"].dtype, object)


def test_forced_columns_dtypes():
    string = """
    customers:
        num_rows: 50
        columns:
            grn_01_str:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
                dtype: str
                seed: 1
            grn_02_str:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: False
                prob_null: 0.5
                dtype: str
                seed: 1
            gdt_01_str:
              type: generate_dates
              start_dt: 2019-01-01
              end_dt: 2020-12-31
              freq: M
              dtype: str
              seed: 1
            gvl_01_float:
              type: generate_values 
              sample_values:
                [True, False]
              dtype: float64
            gvl_02_str:
              type: generate_values 
              sample_values:
                [True, False]
              dtype: str
    """

    parsed_dict = yaml.safe_load(string)
    mock_generator = MockDataGenerator(instructions=parsed_dict)
    mock_generator.generate_all()
    output_dicts = mock_generator.all_dataframes
    assert output_dicts["customers"]["grn_01_str"].iloc[1] == "0.8474337369372327"
    assert output_dicts["customers"]["grn_01_str"].iloc[1] != 0.8474337369372327
    assert output_dicts["customers"]["grn_01_str"].iloc[0] == "None"
    assert isinstance(output_dicts["customers"]["grn_01_str"].dtype, object)
    assert output_dicts["customers"]["grn_02_str"].iloc[0] == "None"
    assert output_dicts["customers"]["grn_02_str"].iloc[1] == "84.89593995678604"
    assert isinstance(output_dicts["customers"]["grn_02_str"].dtype, object)
    assert output_dicts["customers"]["gdt_01_str"].iloc[0] == "2019-01-31 00:00:00"
    assert isinstance(output_dicts["customers"]["gdt_01_str"].dtype, object)
    assert output_dicts["customers"]["gvl_01_float"].iloc[0] == 1
    assert output_dicts["customers"]["gvl_01_float"].dtype == np.float64
    assert output_dicts["customers"]["gvl_02_str"].iloc[0] == "True"
    assert isinstance(output_dicts["customers"]["gvl_02_str"].dtype, object)


def test_integer_and_dtypes_exception():
    with pytest.raises(ValueError) as e_info:
        string = """
        customers:
            num_rows: 50
            columns:
                grn_02_integer:
                    type: generate_random_numbers
                    start_range: 1
                    end_range: 100
                    integer: True
                    dtype: str
                    seed: 1
        """
        parsed_dict = yaml.safe_load(string)
        mock_generator = MockDataGenerator(instructions=parsed_dict)
        mock_generator.generate_all()
    assert str(e_info.value) == "Argument 'integer' cannot be " "set with dtype: str"


def test_dtypes_dummy_exception():
    with pytest.raises(TypeError) as e_info:
        string = """
        customers:
            num_rows: 50
            columns:
                grn_02_integer:
                    type: generate_random_numbers
                    start_range: 1
                    end_range: 100
                    integer: False
                    dtype: dummy
                    seed: 1
        """
        parsed_dict = yaml.safe_load(string)
        mock_generator = MockDataGenerator(instructions=parsed_dict)
        mock_generator.generate_all()
    assert str(e_info.value) == "data type 'dummy' not understood"


def test_integer_dummy_exception():
    with pytest.raises(ValueError) as e_info:
        string = """
        customers:
            num_rows: 50
            columns:
                grn_02_integer:
                    type: generate_random_numbers
                    start_range: 1
                    end_range: 100
                    integer: dummy
                    seed: 1
        """
        parsed_dict = yaml.safe_load(string)
        mock_generator = MockDataGenerator(instructions=parsed_dict)
        mock_generator.generate_all()
    assert (
        str(e_info.value) == "Argument 'integer' must be 'True', 'False' or empty. "
        "It can't be dummy"
    )
