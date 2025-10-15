from copy import deepcopy

import polars as pl
import pytest
from matrix.pipelines.run_comparison.matrix_pairs import (
    MatrixPairs,
)
from polars.testing import assert_frame_equal

DRUGS = pl.DataFrame({"source": [1, 2]})
DISEASES = pl.DataFrame({"target": [1, 2]})
EXCLUSION_SET = pl.DataFrame({"source": [1], "target": [1]})
TEST_SET = {"is_known_positive": pl.DataFrame({"source": [2], "target": [2]})}


@pytest.fixture
def matrix_pairs():
    return MatrixPairs(
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=EXCLUSION_SET,
        test_pairs=TEST_SET,
    )


@pytest.fixture
def matrix_pairs_different_drugs():
    return MatrixPairs(
        drugs_list=pl.DataFrame({"source": [1, 2, 3]}),
        diseases_list=DISEASES,
        exclusion_pairs=EXCLUSION_SET,
        test_pairs=TEST_SET,
    )


@pytest.fixture
def matrix_pairs_different_diseases():
    return MatrixPairs(
        drugs_list=DRUGS,
        diseases_list=pl.DataFrame({"target": [1]}),
        exclusion_pairs=EXCLUSION_SET,
        test_pairs=TEST_SET,
    )


@pytest.fixture
def matrix_pairs_different_exclusion():
    return MatrixPairs(
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=pl.DataFrame({"source": [1], "target": [2]}),
        test_pairs=TEST_SET,
    )


@pytest.fixture
def matrix_pairs_different_test():
    return MatrixPairs(
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=EXCLUSION_SET,
        test_pairs={"is_known_positive": pl.DataFrame({"source": [2, 2], "target": [2, 1]})},
    )


def test_to_lazyframe(matrix_pairs):
    # Given an instance of MatrixPairs
    # When the to_lazyframe method is called
    lazyframe = matrix_pairs.to_lazyframe()

    # Then the output is as expected
    assert isinstance(lazyframe, pl.LazyFrame)
    assert_frame_equal(
        lazyframe,
        pl.LazyFrame({"source": [2, 1, 2], "target": [1, 2, 2], "is_known_positive": [False, False, True]}),
        check_row_order=False,
        check_column_order=False,
    )


def test_is_same_base_matrix(
    matrix_pairs,
    matrix_pairs_different_drugs,
    matrix_pairs_different_diseases,
    matrix_pairs_different_exclusion,
    matrix_pairs_different_test,
):
    # Given several different instance of MatrixPairs with varying data
    # When the is_same_base_matrix method is called
    result_different_drugs = matrix_pairs.is_same_base_matrix(matrix_pairs_different_drugs)
    result_different_diseases = matrix_pairs.is_same_base_matrix(matrix_pairs_different_diseases)
    result_different_exclusion = matrix_pairs.is_same_base_matrix(matrix_pairs_different_exclusion)
    result_different_test = matrix_pairs.is_same_base_matrix(matrix_pairs_different_test)

    # Then the result is false only if the drug or disease list is different
    assert result_different_drugs == False
    assert result_different_diseases == False
    assert result_different_exclusion == True
    assert result_different_test == True


def test_equality_method(
    matrix_pairs,
    matrix_pairs_different_drugs,
    matrix_pairs_different_diseases,
    matrix_pairs_different_exclusion,
    matrix_pairs_different_test,
):
    # Given several different instance of MatrixPairs with varying data as well as a copy with identical data
    matrix_pairs_copy = deepcopy(matrix_pairs)

    # When the equality method is called
    result_copy = matrix_pairs == matrix_pairs_copy
    result_different_drugs = matrix_pairs == matrix_pairs_different_drugs
    result_different_diseases = matrix_pairs == matrix_pairs_different_diseases
    result_different_exclusion = matrix_pairs == matrix_pairs_different_exclusion
    result_different_test = matrix_pairs == matrix_pairs_different_test

    # Then equality is True only for the copy
    assert result_copy == True
    assert result_different_drugs == False
    assert result_different_diseases == False
    assert result_different_exclusion == False
    assert result_different_test == False
