from typing import List
from pytest import fixture
import random
import pytest
from random import choices
from string import ascii_uppercase, digits

from matrix.utils.pair_algorithms import efficient_alternate_pairs

_chars = ascii_uppercase + digits
PAIR_COUNT = 3000
DRUG_COUNT = 100
DISEASE_COUNT = 3000


def _random_strings(count=100, k=8):
    return [str.join("", choices(_chars, k=k)) for c in range(count)]


@fixture()
def drugs():
    return [f"Drug:{d}" for d in _random_strings(DRUG_COUNT)]


@fixture()
def diseases():
    return [f"Disease:{d}" for d in _random_strings(DISEASE_COUNT)]


@fixture()
def pairs(diseases, drugs):
    pair_drugs = choices(drugs, k=PAIR_COUNT)
    pair_diseases = choices(diseases, k=PAIR_COUNT)
    return list(zip(pair_drugs, pair_diseases))


def test_efficient_pairs(
    drugs: List[str], diseases: List[str], pairs: List[tuple[str]]
):
    multi = 10

    pairs_s = set(pairs)
    chosen_drugs = set(x[0] for x in pairs)
    chosen_diseases = set(x[1] for x in pairs)

    results = efficient_alternate_pairs(drugs, diseases, pairs, multi=multi)

    # assert length == multiplier of original length*multi
    assert len(pairs) * multi == len(results)
    # assert no common pairs
    assert len(pairs_s & set(results)) == 0
    # assert each item has 1 element from the original
    for new_pair in results:
        assert (new_pair[0] in chosen_drugs) or (new_pair[1] in chosen_diseases)


def test_efficient_pairs_abort(
    drugs: List[str], diseases: List[str], pairs: List[tuple[str]]
):
    drugs = drugs[0:1]
    diseases = diseases[0:1]
    # starving the algorithm of any new drugs/diseases, it will not find anything new and give up
    with pytest.raises(ValueError) as e:
        efficient_alternate_pairs(drugs, diseases, pairs, multi=100)
