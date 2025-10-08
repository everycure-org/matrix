# NOTE: This file was partially generated using AI assistance.
import random

import pandas as pd
import pytest
from matrix.pipelines.fabricator.pipeline import remove_overlap
from pandas.testing import assert_frame_equal


@pytest.fixture(autouse=True)
def test_seed():
    random.seed(42)
    return 42


def test_remove_overlap_with_overlap():
    # Given: DataFrames with overlapping CURIEs
    drug_list = pd.DataFrame({"translator_id": ["DRUG:1", "DRUG:2", "COMMON:1"], "other_drug_col": [1, 2, 3]})
    disease_list = pd.DataFrame({"translator_id": ["DIS:A", "DIS:B", "COMMON:1"], "other_disease_col": ["a", "b", "c"]})
    expected_drug_list = pd.DataFrame({"translator_id": ["DRUG:1", "DRUG:2"], "other_drug_col": [1, 2]})
    expected_disease_list = pd.DataFrame({"translator_id": ["DIS:A", "DIS:B"], "other_disease_col": ["a", "b"]})

    # When: remove_overlap is called
    result = remove_overlap(
        disease_list=disease_list.copy(), drug_list=drug_list.copy()
    )  # Use copy to avoid modifying originals

    # Then: Overlapping CURIEs are removed from both lists
    assert_frame_equal(result["drug_list"], expected_drug_list, check_dtype=False, check_index_type=False)
    assert_frame_equal(result["disease_list"], expected_disease_list, check_dtype=False, check_index_type=False)


def test_remove_overlap_without_overlap():
    # Given: DataFrames with no overlapping CURIEs
    drug_list = pd.DataFrame({"translator_id": ["DRUG:1", "DRUG:2"], "other_drug_col": [1, 2]})
    disease_list = pd.DataFrame({"translator_id": ["DIS:A", "DIS:B"], "other_disease_col": ["a", "b"]})

    # When: remove_overlap is called
    result = remove_overlap(disease_list=disease_list.copy(), drug_list=drug_list.copy())

    # Then: The original DataFrames are returned unchanged
    assert_frame_equal(result["drug_list"], drug_list, check_dtype=False, check_index_type=False)
    assert_frame_equal(result["disease_list"], disease_list, check_dtype=False, check_index_type=False)


def test_remove_overlap_empty_inputs():
    # Given: Empty DataFrames
    empty_drug_list = pd.DataFrame(columns=["translator_id", "other_drug_col"])
    empty_disease_list = pd.DataFrame(columns=["translator_id", "other_disease_col"])

    # When: remove_overlap is called with two empty lists
    result_both_empty = remove_overlap(disease_list=empty_disease_list.copy(), drug_list=empty_drug_list.copy())
    # Then: Empty DataFrames are returned
    assert_frame_equal(result_both_empty["drug_list"], empty_drug_list, check_dtype=False, check_index_type=False)
    assert_frame_equal(result_both_empty["disease_list"], empty_disease_list, check_dtype=False, check_index_type=False)

    # Given: One empty DataFrame and one non-empty
    drug_list = pd.DataFrame({"translator_id": ["DRUG:1", "DRUG:2"], "other_drug_col": [1, 2]})
    # When: remove_overlap is called with one empty list
    result_one_empty = remove_overlap(disease_list=empty_disease_list.copy(), drug_list=drug_list.copy())
    # Then: The original DataFrames are returned (one empty, one not)
    assert_frame_equal(result_one_empty["drug_list"], drug_list, check_dtype=False, check_index_type=False)
    assert_frame_equal(result_one_empty["disease_list"], empty_disease_list, check_dtype=False, check_index_type=False)

    # Given: One empty DataFrame and one non-empty (other way round)
    disease_list = pd.DataFrame({"translator_id": ["DIS:A", "DIS:B"], "other_disease_col": ["a", "b"]})
    # When: remove_overlap is called with the other list empty
    result_other_empty = remove_overlap(disease_list=disease_list.copy(), drug_list=empty_drug_list.copy())
    # Then: The original DataFrames are returned
    assert_frame_equal(result_other_empty["drug_list"], empty_drug_list, check_dtype=False, check_index_type=False)
    assert_frame_equal(result_other_empty["disease_list"], disease_list, check_dtype=False, check_index_type=False)
