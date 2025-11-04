from copy import deepcopy

import polars as pl
import pytest
from matrix.pipelines.run_comparison.matrix_pairs import (
    MatrixPairs,
    give_matrix_pairs_from_lazyframe,
    harmonize_matrix_pairs,
)
from matrix.pipelines.run_comparison.nodes import (
    check_base_matrices_consistent,
    harmonize_matrix_pairs,
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


@pytest.fixture
def matrix_lazyframe():
    return pl.LazyFrame({"source": [2, 1, 2], "target": [1, 2, 2], "is_known_positive": [False, False, True]})


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


def test_equality_method(
    matrix_pairs,
    matrix_pairs_different_drugs,
    matrix_pairs_different_diseases,
    matrix_pairs_different_exclusion,
    matrix_pairs_different_test,
):
    # Given several different instance of MatrixPairs with varying data as well as a copy with identical data
    matrix_pairs_copy = deepcopy(matrix_pairs)
    # When the harmonize method is called
    result_copy = matrix_pairs.harmonize(matrix_pairs_copy)
    result_different_drugs = matrix_pairs.harmonize(matrix_pairs_different_drugs)
    result_different_diseases = matrix_pairs.harmonize(matrix_pairs_different_diseases)
    result_different_exclusion = matrix_pairs.harmonize(matrix_pairs_different_exclusion)
    result_different_test = matrix_pairs.harmonize(matrix_pairs_different_test)
    result_different_test_no_exclusion = matrix_pairs.harmonize(
        matrix_pairs_different_test, exclude_inconsistent_pairs=False
    )

    # Then the result is as expected
    assert result_copy == matrix_pairs
    assert result_different_drugs == matrix_pairs  # Extra drug on matrix_pairs_different_drugs gets dropped
    assert result_different_diseases == matrix_pairs_different_diseases  # Extra disease in matrix_pairs gets dropped
    assert result_different_exclusion == MatrixPairs(  # Union of exclusion sets
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=pl.DataFrame({"source": [1, 1], "target": [1, 2]}),
        test_pairs=TEST_SET,
    )
    assert result_different_test == MatrixPairs(  # Intersection of test sets. Inconsistent pair dropped.
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=pl.DataFrame({"source": [1, 2], "target": [1, 1]}),
        test_pairs=TEST_SET,
    )
    assert (  # Intersection of test sets. Inconsistent pair not dropped.
        result_different_test_no_exclusion == matrix_pairs
    )


def test_give_matrix_pairs_from_lazyframe(matrix_pairs, matrix_lazyframe):
    # Given a Polars LazyFrame representing the matrix pairs
    # When generating an instance of the MatrixPairs object
    matrix_pairs_generated = give_matrix_pairs_from_lazyframe(
        matrix_lazyframe, available_ground_truth_cols=["is_known_positive"]
    )
    # Then the result is as expected
    assert matrix_pairs_generated == matrix_pairs


def test_harmonize_matrix_pairs(
    matrix_pairs,
    matrix_pairs_different_drugs,
    matrix_pairs_different_exclusion,
    matrix_pairs_different_test,
):
    # Given several instance of MatrixPairs with varying data
    # When calling the harmonize_matrix_pairs function
    harmonized_matrix_pairs = harmonize_matrix_pairs(
        matrix_pairs,
        matrix_pairs_different_drugs,
        matrix_pairs_different_exclusion,
        matrix_pairs_different_test,
        exclude_inconsistent_pairs=False,
    )

    # Then the result is as expected
    assert harmonized_matrix_pairs == MatrixPairs(
        drugs_list=DRUGS,
        diseases_list=DISEASES,
        exclusion_pairs=pl.DataFrame({"source": [1, 1], "target": [1, 2]}),
        test_pairs=TEST_SET,
    )


def test_check_base_matrices_consistent(
    matrix_pairs,
    matrix_pairs_different_drugs,
    matrix_pairs_different_exclusion,
    matrix_pairs_different_test,
):
    # Given several instance of MatrixPairs with varying data
    # When calling the check_base_matrices_consistent function on matrices with different drugs list
    result_different_drugs = check_base_matrices_consistent(
        matrix_pairs,
        matrix_pairs_different_drugs,
        matrix_pairs_different_exclusion,
        matrix_pairs_different_test,
    )
    # Or the same drug and disease lists
    result_same_drugs = check_base_matrices_consistent(
        matrix_pairs,
        matrix_pairs_different_exclusion,
        matrix_pairs_different_test,
    )

    # Then the result is as expected
    assert result_different_drugs == False
    assert result_same_drugs == True


def check_matrix_pairs_equal(matrix_pairs, matrix_pairs_different_drugs):
    # Given three identical MatrixPairs objects and a single different one
    matrix_pairs_copy_1 = deepcopy(matrix_pairs)
    matrix_pairs_copy_2 = deepcopy(matrix_pairs)
    # When calling the check_matrix_pairs_equal function
    result_equal = check_matrix_pairs_equal(
        matrix_pairs,
        matrix_pairs_copy_1,
        matrix_pairs_copy_2,
    )
    result_different = check_matrix_pairs_equal(
        matrix_pairs,
        matrix_pairs_copy_1,
        matrix_pairs_different_drugs,
    )
    # Then the result is as expected
    assert result_equal == True
    assert result_different == False
