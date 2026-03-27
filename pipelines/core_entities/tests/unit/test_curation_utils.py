import numpy as np

import pandas as pd
import pytest

# NOTE: This file was partially generated using AI assistance.
from core_entities.utils.curation_utils import (
    apply_patch,
    create_search_term_from_curated_drug_list,
    filter_dataframe_by_columns,
)


# ---- Fixtures ----
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "name": ["a", "b", "c", "d"],
            "value": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def sample_patch_df():
    return pd.DataFrame({"id": ["2", "3"], "name": ["b_updated", "c_updated"], "value": [25, 35]})


# ---- Tests for apply_patch ----
def test_apply_patch_basic(sample_df, sample_patch_df):
    result = apply_patch(sample_df, sample_patch_df, ["name", "value"], "id")

    expected = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "name": ["a", "b_updated", "c_updated", "d"],
            "value": [10, 25, 35, 40],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_apply_patch_with_nonexistent_column():
    df = pd.DataFrame({"id": ["1", "2"], "name": ["a", "b"]})
    patch_df = pd.DataFrame({"id": ["1"], "name": ["a_updated"], "new_col": ["new_value"]})

    with pytest.raises(ValueError, match="Patch column new_col not found in df"):
        apply_patch(df, patch_df, ["name", "new_col"], "id")


class TestFilterDataframeByColumns:
    def test_filters_rows_matching_single_column(self) -> None:
        # Given: a DataFrame with a status column containing mixed values
        df = pd.DataFrame(
            {
                "status": ["APPROVED", "DELETED", "APPROVED", "PENDING"],
                "name": ["DrugA", "DrugB", "DrugC", "DrugD"],
            }
        )
        filter_columns = {"status": "APPROVED"}

        # When: we filter by status == APPROVED
        result = filter_dataframe_by_columns(df, filter_columns)

        # Then: only rows with APPROVED status remain
        expected = pd.DataFrame(
            {
                "status": ["APPROVED", "APPROVED"],
                "name": ["DrugA", "DrugC"],
            },
            index=[0, 2],
        )
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), result.reset_index(drop=True))

    def test_filters_rows_matching_multiple_columns(self) -> None:
        # Given: a DataFrame with status and region columns
        df = pd.DataFrame(
            {
                "status": ["APPROVED", "APPROVED", "DELETED"],
                "region": ["USA", "EU", "USA"],
                "name": ["DrugA", "DrugB", "DrugC"],
            }
        )
        filter_columns = {"status": "APPROVED", "region": "USA"}

        # When: we filter by both status and region
        result = filter_dataframe_by_columns(df, filter_columns)

        # Then: only rows matching both conditions remain
        expected = pd.DataFrame(
            {
                "status": ["APPROVED"],
                "region": ["USA"],
                "name": ["DrugA"],
            },
            index=[0],
        )
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), result.reset_index(drop=True))

    def test_returns_empty_dataframe_when_no_rows_match(self) -> None:
        # Given: a DataFrame where no rows match the filter
        df = pd.DataFrame(
            {
                "status": ["DELETED", "PENDING"],
                "name": ["DrugA", "DrugB"],
            }
        )
        filter_columns = {"status": "APPROVED"}

        # When: we filter by status == APPROVED
        result = filter_dataframe_by_columns(df, filter_columns)

        # Then: result is empty
        assert len(result) == 0

    def test_returns_all_rows_when_filter_columns_is_empty(self) -> None:
        # Given: a DataFrame and an empty filter_columns dict
        df = pd.DataFrame(
            {
                "status": ["APPROVED", "DELETED"],
                "name": ["DrugA", "DrugB"],
            }
        )
        filter_columns = {}

        # When: we filter with no filter columns
        result = filter_dataframe_by_columns(df, filter_columns)

        # Then: all rows are returned unchanged
        assert len(df) == len(result)

    def test_warns_and_skips_missing_filter_column(self, caplog: pytest.LogCaptureFixture) -> None:
        # Given: a DataFrame without the filter column
        df = pd.DataFrame(
            {
                "name": ["DrugA", "DrugB"],
            }
        )
        filter_columns = {"nonexistent_col": "APPROVED"}

        # When: we filter with a column that does not exist
        with caplog.at_level("WARNING"):
            result = filter_dataframe_by_columns(df, filter_columns)

        # Then: a warning is logged and all rows are returned
        assert any("nonexistent_col" in message for message in caplog.messages)
        assert len(df) == len(result)

    def test_does_not_modify_original_dataframe(self) -> None:
        # Given: a DataFrame and a filter
        df = pd.DataFrame(
            {
                "status": ["APPROVED", "DELETED"],
                "name": ["DrugA", "DrugB"],
            }
        )
        original_length = len(df)
        filter_columns = {"status": "APPROVED"}

        # When: we filter the DataFrame
        filter_dataframe_by_columns(df, filter_columns)

        # Then: the original DataFrame is unchanged
        assert original_length == len(df)

    def test_filters_correctly_with_all_rows_matching(self) -> None:
        # Given: a DataFrame where all rows match the filter
        df = pd.DataFrame(
            {
                "status": ["APPROVED", "APPROVED"],
                "name": ["DrugA", "DrugB"],
            }
        )
        filter_columns = {"status": "APPROVED"}

        # When: we filter by status == APPROVED
        result = filter_dataframe_by_columns(df, filter_columns)

        # Then: all rows are returned
        assert len(df) == len(result)


def _make_drug_df(**kwargs) -> pd.DataFrame:
    base = {
        "name": ["DrugA"],
        "id": ["drug_001"],
        "available_in_combo_with": [[]],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


class TestCreateSearchTermFromCuratedDrugList:
    def test_returns_dataframe(self) -> None:
        # Given: a minimal curated drug list with one string column
        df = _make_drug_df(brand_name=["Aspirin"])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: result is a DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_output_contains_required_columns(self) -> None:
        # Given: a minimal curated drug list
        df = _make_drug_df(brand_name=["Aspirin"])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: as a result has all required columns
        assert set(["name", "id", "search_terms", "available_in_combo_with"]).issubset(result.columns)

    def test_string_column_adds_term_to_search_terms(self) -> None:
        # Given: a drug with a non-empty string column
        df = _make_drug_df(brand_name=["Aspirin"])
        # When: we create search terms using that column
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: the string value appears in search_terms
        assert {"Aspirin"} == result.iloc[0]["search_terms"]

    def test_list_column_adds_all_string_elements(self) -> None:
        # Given: a drug with a list of synonyms
        df = _make_drug_df(synonyms=[["Aspirin", "ASA"]])
        # When: we create search terms using the list column
        result = create_search_term_from_curated_drug_list(df, ["synonyms"])
        # Then: all string elements from the list are in search_terms
        assert {"Aspirin", "ASA"} == result.iloc[0]["search_terms"]

    def test_numpy_array_column_adds_all_string_elements(self) -> None:
        # Given: a drug with a numpy array of synonyms
        df = _make_drug_df(synonyms=[np.array(["Aspirin", "ASA"])])
        # When: we create search terms using the array column
        result = create_search_term_from_curated_drug_list(df, ["synonyms"])
        # Then: all string elements from the array are in search_terms
        assert {"Aspirin", "ASA"} == result.iloc[0]["search_terms"]

    def test_empty_string_is_excluded_from_search_terms(self) -> None:
        # Given: a drug with an empty string in a column
        df = _make_drug_df(brand_name=[""])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: the empty string is not in search_terms
        assert set() == result.iloc[0]["search_terms"]

    def test_empty_string_in_list_is_excluded_from_search_terms(self) -> None:
        # Given: a drug list column containing an empty string among valid values
        df = _make_drug_df(synonyms=[["Aspirin", "", "ASA"]])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["synonyms"])
        # Then: the empty string is excluded but valid values remain
        assert {"Aspirin", "ASA"} == result.iloc[0]["search_terms"]

    def test_non_string_elements_in_list_are_excluded(self) -> None:
        # Given: a drug list column with mixed types
        df = _make_drug_df(synonyms=[["Aspirin", 42, None]])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["synonyms"])
        # Then: only string elements are included
        assert {"Aspirin"} == result.iloc[0]["search_terms"]

    def test_multiple_columns_are_combined_into_search_terms(self) -> None:
        # Given: a drug with values in two separate matching columns
        df = _make_drug_df(brand_name=["Aspirin"], generic_name=["Acetylsalicylic Acid"])
        # When: we create search terms using both columns
        result = create_search_term_from_curated_drug_list(df, ["brand_name", "generic_name"])
        # Then: values from both columns appear in search_terms
        assert {
            "Aspirin",
            "Acetylsalicylic Acid",
            "Acetylsalicylic, Acid",
            "Acid, Acetylsalicylic",
        } == result.iloc[0]["search_terms"]

    def test_two_word_term_adds_comma_order_variants(self) -> None:
        # Given: a two-word matching term
        df = _make_drug_df(brand_name=["conjugated estrogens"])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: original and comma-separated word-order variants are present
        assert {
            "conjugated estrogens",
            "conjugated, estrogens",
            "estrogens, conjugated",
        } == result.iloc[0]["search_terms"]

    def test_empty_drug_list_returns_empty_dataframe(self) -> None:
        # Given: an empty curated drug list
        df = pd.DataFrame(columns=["name", "id", "available_in_combo_with", "brand_name"])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: the result is an empty DataFrame
        assert 0 == len(result)

    def test_output_row_count_matches_input_row_count(self) -> None:
        # Given: a drug list with multiple rows
        df = pd.DataFrame(
            {
                "name": ["DrugA", "DrugB", "DrugC"],
                "id": ["001", "002", "003"],
                "available_in_combo_with": [[], [], []],
                "brand_name": ["Alpha", "Beta", "Gamma"],
            }
        )
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: output has the same number of rows as input
        assert 3 == len(result)

    def test_name_and_id_are_preserved_in_output(self) -> None:
        # Given: a drug with known name and id
        df = _make_drug_df(brand_name=["Aspirin"])
        df["name"] = "DrugA"
        df["id"] = "drug_001"
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: name and id are preserved in the output row
        assert "DrugA" == result.iloc[0]["name"]
        assert "drug_001" == result.iloc[0]["id"]

    def test_available_in_combo_with_is_preserved_in_output(self) -> None:
        # Given: a drug with a non-empty available_in_combo_with value
        df = _make_drug_df(brand_name=["Aspirin"])
        df["available_in_combo_with"] = [["DrugB"]]
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: available_in_combo_with is preserved
        assert ["DrugB"] == result.iloc[0]["available_in_combo_with"]

    def test_no_matching_columns_yields_empty_search_terms(self) -> None:
        # Given: a drug list where the specified column has only empty strings
        df = _make_drug_df(brand_name=[""])
        # When: we create search terms
        result = create_search_term_from_curated_drug_list(df, ["brand_name"])
        # Then: search_terms is an empty set
        assert set() == result.iloc[0]["search_terms"]

    def test_duplicate_values_across_columns_appear_once_in_search_terms(self) -> None:
        # Given: a drug where the same term appears in two columns
        df = _make_drug_df(brand_name=["Aspirin"], generic_name=["Aspirin"])
        # When: we create search terms from both columns
        result = create_search_term_from_curated_drug_list(df, ["brand_name", "generic_name"])
        # Then: the duplicate term appears only once (set semantics)
        assert {"Aspirin"} == result.iloc[0]["search_terms"]
