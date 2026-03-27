import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.pipelines.drug_list.nodes import (
    resolve_fda_drugs_matches_to_drug_list_filtered,
    resolve_fda_drugs_that_are_biosimilar_and_are_generic,
)
from core_entities.utils.fda_drugs_utils import (
    _active_ingredients_match_combo,
    _any_substring_match,
    _as_list,
    _extract_active_ingredient_names,
    _extract_strings_from_nested_value,
    _extract_values_for_path,
    _find_matching_fda_rows,
    _is_anda_application_number,
    _is_anda_or_bla_application_number,
    _is_likely_drug_like_combo_part,
    _is_probable_combo_expression,
    _matches_any_search_term,
    _normalize_string_terms,
    _split_combo_expression_parts,
    _unique_non_empty_strings,
    extract_openfda_field,
    extract_product_active_ingredients,
    extract_product_marketing_status,
    filter_fda_rows,
    has_anda_application_number,
    match_drug_to_fda_worker,
)


def test_find_matching_fda_rows_uses_configured_openfda_paths() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "ANDA111111",
                "openfda": {
                    "brand_name": np.array(["Brand A"], dtype=object),
                    "generic_name": np.array(["alpha"], dtype=object),
                },
                "products": [{"active_ingredients": [{"name": "alpha"}]}],
            },
            {
                "application_number": "ANDA222222",
                "openfda": {
                    "brand_name": np.array(["Brand B"], dtype=object),
                    "generic_name": np.array(["beta"], dtype=object),
                },
                "products": [{"active_ingredients": [{"name": "beta"}]}],
            },
        ]
    )

    matches = _find_matching_fda_rows(
        search_terms={"brand a"},
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["openfda.brand_name"],
    )

    assert len(matches) == 1
    assert matches[0]["application_number"] == "ANDA111111"


def test_find_matching_fda_rows_supports_products_active_ingredients_path() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "BLA123456",
                "openfda": {"brand_name": np.array(["Combo Drug"], dtype=object)},
                "products": np.array(
                    [
                        {
                            "active_ingredients": np.array(
                                [
                                    {"name": "drug x"},
                                    {"name": "helper y sulfate"},
                                ],
                                dtype=object,
                            )
                        }
                    ],
                    dtype=object,
                ),
            }
        ]
    )

    matches = _find_matching_fda_rows(
        search_terms={"helper y"},
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["products.active_ingredients.name"],
    )

    assert len(matches) == 1
    assert matches[0]["application_number"] == "BLA123456"


def test_find_matching_fda_rows_ignores_non_string_and_blank_column_paths() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "ANDA101010",
                "openfda": {"brand_name": ["Brand A"]},
                "products": [],
            }
        ]
    )

    matches = _find_matching_fda_rows(
        search_terms={"brand a"},
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=[123, "   ", "openfda.brand_name"],  # type: ignore[list-item]
    )

    assert len(matches) == 1
    assert matches[0]["application_number"] == "ANDA101010"


def test_filter_fda_rows_returns_match_for_single_ingredient_generic() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA123456",
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "acetaminophen"}],
                        }
                    ],
                }
            ],
            "search_terms": ["acetaminophen"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["application_number"] == "ANDA123456"


def test_filter_fda_rows_returns_match_for_valid_combo_partner() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "BLA654321",
                    "products": [
                        {
                            "marketing_status": "over-the-counter",
                            "active_ingredients": [
                                {"name": "drug a"},
                                {"name": "drug b"},
                            ],
                        }
                    ],
                }
            ],
            "search_terms": ["drug a"],
            "available_in_combo_with": ["drug b"],
        }
    )

    result = filter_fda_rows(row)

    assert len(result) == 1
    assert result[0]["application_number"] == "BLA654321"


def test_filter_fda_rows_does_not_match_other_single_ingredient_products() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA777777",
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "minoxidil"}],
                        },
                        {
                            "marketing_status": "Over-the-counter",
                            "active_ingredients": [{"name": "minoxidil"}, {"name": "ascorbic acid"}],
                        },
                    ],
                }
            ],
            "search_terms": ["ascorbic acid", "vitamin c"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


# FDA API sometimes returns multi-ingredient products as a single active ingredient with an "and" in the name, which should not be considered a match for either ingredient alone.
# as we are using substring matching to find candidate FDA rows, we need to ensure that we are not including these combo expressions as matches for the individual ingredients.
# This ensure substring searching for a drug without we mentioning the salt (like abecivir in our system should match with abecivir hydrochloride but not with a combo product that has "abecivir and something else" as the active ingredient name).
def test_filter_fda_rows_excludes_single_ingredient_combo_expression_with_and() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "BLA123456",
                    "openfda": {
                        "substance_name": ["amivantamab and hyaluronidase-lpuj"],
                    },
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "amivantamab and hyaluronidase-lpuj"}],
                        }
                    ],
                }
            ],
            "search_terms": ["amivantamab"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_excludes_single_ingredient_combo_expression_with_ampersand() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA123456",
                    "products": [
                        {
                            "marketing_status": "Over-the-counter",
                            "active_ingredients": [{"name": "drug a & drug b"}],
                        }
                    ],
                }
            ],
            "search_terms": ["drug a"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_excludes_single_ingredient_combo_expression_with_comma() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA888888",
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "drug a, drug b"}],
                        }
                    ],
                }
            ],
            "search_terms": ["drug a"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_excludes_single_ingredient_combo_expression_with_semicolon() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA777778",
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "drug a; drug b"}],
                        }
                    ],
                }
            ],
            "search_terms": ["drug a"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_excludes_row_when_substance_name_is_combo_expression() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA999999",
                    "openfda": {
                        "substance_name": ["amivantamab and hyaluronidase-lpuj"],
                    },
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "amivantamab"}],
                        }
                    ],
                }
            ],
            "search_terms": ["amivantamab"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_keeps_legitimate_name_when_full_phrase_is_search_term() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                {
                    "application_number": "ANDA765432",
                    "products": [
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": [{"name": "sodium and magnesium hydroxide"}],
                        }
                    ],
                }
            ],
            "search_terms": ["sodium and magnesium hydroxide"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert len(result) == 1
    assert result[0]["application_number"] == "ANDA765432"


def test_filter_fda_rows_handles_numpy_array_active_ingredients() -> None:
    row = pd.Series(
        {
            "fda_rows": np.array(
                [
                    {
                        "application_number": "ANDA246810",
                        "products": np.array(
                            [
                                {
                                    "marketing_status": "Prescription",
                                    "active_ingredients": np.array(
                                        [{"name": "acetaminophen"}],
                                        dtype=object,
                                    ),
                                }
                            ],
                            dtype=object,
                        ),
                    }
                ],
                dtype=object,
            ),
            "search_terms": np.array(["acetaminophen"], dtype=object),
            "available_in_combo_with": np.array([], dtype=object),
        }
    )

    result = filter_fda_rows(row)

    assert len(result) == 1
    assert result[0]["application_number"] == "ANDA246810"


def test_filter_fda_rows_returns_empty_when_fda_rows_is_none() -> None:
    row = pd.Series(
        {
            "fda_rows": None,
            "search_terms": ["acetaminophen"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_filter_fda_rows_skips_non_dict_rows_and_invalid_product_shapes() -> None:
    row = pd.Series(
        {
            "fda_rows": [
                "not-a-dict",
                {
                    "application_number": "ANDA111111",
                    "products": None,
                },
                {
                    "application_number": "ANDA222222",
                    "products": [
                        "bad-product",
                        {
                            "marketing_status": "Prescription",
                            "active_ingredients": None,
                        },
                    ],
                },
            ],
            "search_terms": ["acetaminophen"],
            "available_in_combo_with": [],
        }
    )

    result = filter_fda_rows(row)

    assert result == []


def test_resolve_filtered_adds_derived_columns_without_embedded_series_objects() -> None:
    df = pd.DataFrame(
        {
            "fda_rows": [
                [
                    {
                        "application_number": "BLA111111",
                        "openfda": {
                            "brand_name": ["Brand X", "Brand X"],
                            "generic_name": ["drug x"],
                            "substance_name": ["drug x hydrochloride"],
                        },
                        "products": [
                            {
                                "marketing_status": "Prescription",
                                "active_ingredients": [{"name": "drug x"}],
                            },
                            {
                                "marketing_status": "Over-the-counter",
                                "active_ingredients": [{"name": "drug x"}, {"name": "helper y"}],
                            },
                        ],
                    },
                    {
                        "application_number": "ANDA333333",
                        "openfda": {
                            "brand_name": ["Brand X Legacy"],
                            "generic_name": ["drug x legacy"],
                            "substance_name": ["drug x legacy salt"],
                        },
                        "products": [
                            {
                                "marketing_status": "Discontinued",
                                "active_ingredients": [{"name": "drug x"}],
                            }
                        ],
                    },
                ],
                [
                    {
                        "application_number": "NDA222222",
                        "openfda": {
                            "brand_name": ["Brand Y"],
                            "generic_name": ["drug y"],
                            "substance_name": ["drug y hydrochloride"],
                        },
                        "products": [
                            {
                                "marketing_status": "Prescription",
                                "active_ingredients": [{"name": "drug y"}],
                            }
                        ],
                    }
                ],
            ],
            "fda_match_count": [1, 1],
            "drug_name": ["drug x", "drug y"],
            "id": ["EC:00001", "EC:00002"],
            "search_terms": [["drug x"], ["drug y"]],
            "available_in_combo_with": [[], []],
        }
    )

    result_parquet, result_tsv = resolve_fda_drugs_matches_to_drug_list_filtered(df)

    assert "filtered_fda_values" in result_parquet.columns
    assert "fda_rows" in result_parquet.columns
    assert "filtered_fda_values_count" in result_parquet.columns
    assert "is_fda_generic_drug" in result_parquet.columns
    assert "is_biologics" in result_parquet.columns
    assert "brand_name" in result_parquet.columns
    assert "generic_name" in result_parquet.columns
    assert "substance_name" in result_parquet.columns
    assert "active_ingredients" in result_parquet.columns
    assert "marketing_status" in result_parquet.columns
    assert "is_anda" in result_parquet.columns

    assert "filtered_fda_values" not in result_tsv.columns
    assert "fda_rows" not in result_tsv.columns
    assert "is_anda" in result_tsv.columns

    assert result_parquet["filtered_fda_values_count"].tolist() == [1, 0]
    assert result_parquet["is_fda_generic_drug"].tolist() == [True, False]
    assert result_parquet["is_biologics"].tolist() == [True, False]
    assert result_parquet["brand_name"].tolist() == [["Brand X"], []]
    assert result_parquet["generic_name"].tolist() == [["drug x"], []]
    assert result_parquet["substance_name"].tolist() == [["drug x hydrochloride"], []]
    assert result_parquet["active_ingredients"].tolist() == [["drug x"], []]
    assert result_parquet["marketing_status"].tolist() == [["Prescription"], []]
    assert result_parquet["is_anda"].tolist() == [False, False]
    assert result_tsv["brand_name"].tolist() == [["Brand X"], []]
    assert result_tsv["is_anda"].tolist() == [False, False]

    assert result_parquet["filtered_fda_values"].map(lambda v: isinstance(v, list)).all()
    contains_series_value = result_parquet.apply(lambda col: col.map(lambda v: isinstance(v, pd.Series)).any()).any()
    assert not contains_series_value


def test_resolve_biosimilar_generic_overrides_only_biosimilars_using_configured_bla_types() -> None:
    df = pd.DataFrame(
        {
            "id": ["EC:00001", "EC:00002", "EC:00003"],
            "drug_name": ["drug a", "drug b", "drug c"],
            "is_biologics": [True, True, False],
            "is_fda_generic_drug": [True, True, False],
            "filtered_fda_values": [
                [{"application_number": "BLA111111", "bla_type": "351(k) Biosimilar"}],
                [{"application_number": "BLA222222", "bla_type": "351(a)"}],
                [{"application_number": "NDA333333"}],
            ],
            "fda_rows": [[], [], []],
        }
    )

    params = {"generic_status_bla_type": ["351(k) Interchangeable", "351(k) Biosimilar"]}

    result_parquet, result_tsv = resolve_fda_drugs_that_are_biosimilar_and_are_generic(
        df,
        params,
        pd.DataFrame(),
    )

    assert result_parquet["is_fda_generic_drug"].tolist() == [True, False, False]
    assert result_parquet["biosimilar_bla_types"].tolist() == [["351(k) Biosimilar"], ["351(a)"], []]

    assert "fda_rows" in result_parquet.columns
    assert "filtered_fda_values" in result_parquet.columns
    assert "fda_rows" not in result_tsv.columns
    assert "filtered_fda_values" not in result_tsv.columns


def test_resolve_biosimilar_generic_supports_nested_param_shape() -> None:
    df = pd.DataFrame(
        {
            "id": ["EC:00001"],
            "drug_name": ["drug a"],
            "is_biologics": [True],
            "is_fda_generic_drug": [False],
            "filtered_fda_values": [[{"application_number": "BLA111111", "bla_type": "351(k) Interchangeable"}]],
            "fda_rows": [[]],
        }
    )

    params = {"fda_purple_book": {"generic_status_bla_type": ["351(k) Interchangeable"]}}

    result_parquet, _ = resolve_fda_drugs_that_are_biosimilar_and_are_generic(
        df,
        params,
        pd.DataFrame(),
    )

    assert result_parquet["is_fda_generic_drug"].tolist() == [True]


def test_resolve_biosimilar_generic_uses_purple_book_bla_number_lookup() -> None:
    df = pd.DataFrame(
        {
            "id": ["EC:00001"],
            "drug_name": ["adalimumab"],
            "is_biologics": [True],
            "is_fda_generic_drug": [False],
            "filtered_fda_values": [[{"application_number": "bla761071"}]],
            "fda_rows": [[]],
        }
    )

    purple_book_df = pd.DataFrame(
        {
            "BLA Number": [761071],
            "BLA Type": ["351(k) Interchangeable"],
            "Proper Name": ["adalimumab"],
        }
    )
    params = {"generic_status_bla_type": ["351(k) Interchangeable"]}

    result_parquet, _ = resolve_fda_drugs_that_are_biosimilar_and_are_generic(df, params, purple_book_df)

    assert result_parquet["biosimilar_application_numbers"].tolist() == [["bla761071"]]
    assert result_parquet["biosimilar_bla_types"].tolist() == [["351(k) Interchangeable"]]
    assert result_parquet["is_fda_generic_drug"].tolist() == [True]


# Tests for _normalize_string_terms — covers behaviour before refactoring away inline duplication.


def test_normalize_string_terms_lowercases_and_strips() -> None:
    # Given: values with mixed case and surrounding whitespace
    # When: normalized
    result = _normalize_string_terms(["  Acebutolol ", "METFORMIN"])
    # Then: all values are lowercased and stripped
    assert result == ["acebutolol", "metformin"]


def test_normalize_string_terms_skips_empty_strings() -> None:
    # Given: a mix of valid and blank strings
    result = _normalize_string_terms(["aspirin", "", "   ", "ibuprofen"])
    # Then: only non-empty strings are returned
    assert result == ["aspirin", "ibuprofen"]


def test_normalize_string_terms_skips_non_strings() -> None:
    # Given: list containing non-string values (e.g. None, int)
    result = _normalize_string_terms(["aspirin", None, 42, "ibuprofen"])  # type: ignore[list-item]
    # Then: non-strings are silently skipped
    assert result == ["aspirin", "ibuprofen"]


def test_normalize_string_terms_returns_empty_for_none_input() -> None:
    # Given: None passed as the values argument
    result = _normalize_string_terms(None)  # type: ignore[arg-type]
    # Then: an empty list is returned without error
    assert result == []


def test_normalize_string_terms_accepts_single_string_input() -> None:
    result = _normalize_string_terms("  Aspirin  ")
    assert result == ["aspirin"]


def test_normalize_string_terms_accepts_numpy_array_input() -> None:
    # Given: a numpy array input as produced by parquet/list-like roundtrips
    result = _normalize_string_terms(np.array([" A ", "B"], dtype=object))
    # Then: terms are normalized without array truthiness errors
    assert result == ["a", "b"]


def test_split_combo_expression_parts_empty_string_returns_empty_list() -> None:
    assert _split_combo_expression_parts("   ") == []


# The Problem:


# FDA sometimes encodes a combo drug (two drugs together) as a single active_ingredients field with a separator: "amivantamab & hyaluronidase-lpuj"
# When we search for just "amivantamab" (single drug), substring matching would incorrectly find this combo product
# This is a false positive—you wanted a single-ingredient product, not a combination
# Why the Test:
# By verifying that _is_probable_combo_expression() returns True for "amivantamab & hyaluronidase-lpuj", we ensure the filtering code in _filter_fda_rows() can reject this product from matching our single-drug search.
def test_is_likely_drug_like_combo_part_rejects_too_short_values() -> None:
    assert _is_likely_drug_like_combo_part("ab") is False


def test_is_likely_drug_like_combo_part_returns_true_for_valid_drug_like_text() -> None:
    assert _is_likely_drug_like_combo_part("drug a") is True


def test_is_probable_combo_expression_returns_false_for_empty_value() -> None:
    assert _is_probable_combo_expression("   ", ["drug a"]) is False


def test_is_probable_combo_expression_returns_false_for_single_part_after_split() -> None:
    assert _is_probable_combo_expression("drug and", ["drug"]) is False


def test_is_probable_combo_expression_returns_false_for_duplicate_parts() -> None:
    assert _is_probable_combo_expression("drug a and drug a", ["drug a"]) is False


def test_is_probable_combo_expression_returns_false_for_non_drug_like_parts() -> None:
    assert _is_probable_combo_expression("drug a and --", ["drug a"]) is False


def test_is_probable_combo_expression_returns_true_for_two_distinct_drug_like_parts() -> None:
    assert _is_probable_combo_expression("drug a and drug b", ["drug a"]) is True


def test_is_probable_combo_expression_returns_true_for_ampersand_separator() -> None:
    assert _is_probable_combo_expression("amivantamab & hyaluronidase-lpuj", ["amivantamab"]) is True


# Tests for _extract_active_ingredient_names — covers its own normalization behaviour.


def test_extract_active_ingredient_names_from_dict_list() -> None:
    # Given: a standard list of ingredient dicts
    ingredients = [{"name": "  Warfarin "}, {"name": "ASPIRIN"}]
    # When: names are extracted
    result = _extract_active_ingredient_names(ingredients)
    # Then: names are lowercased and stripped
    assert result == ["warfarin", "aspirin"]


def test_extract_active_ingredient_names_handles_bare_strings() -> None:
    # Given: ingredients provided as plain strings (alternative schema)
    result = _extract_active_ingredient_names(["Lisinopril", "  Amlodipine  "])
    # Then: strings are normalised the same way as dict entries
    assert result == ["lisinopril", "amlodipine"]


def test_extract_active_ingredient_names_skips_missing_name_key() -> None:
    # Given: a dict without a "name" key
    result = _extract_active_ingredient_names([{"strength": "10mg"}, {"name": "metoprolol"}])
    # Then: only the entry with a name is returned
    assert result == ["metoprolol"]


def test_extract_active_ingredient_names_skips_non_string_name_values() -> None:
    # Given: a dict where "name" is not a string
    result = _extract_active_ingredient_names([{"name": 42}, {"name": "atorvastatin"}])
    # Then: the non-string value is silently skipped
    assert result == ["atorvastatin"]


def test_extract_active_ingredient_names_skips_empty_names() -> None:
    # Given: a dict where "name" is an empty/blank string
    result = _extract_active_ingredient_names([{"name": "  "}, {"name": "ramipril"}])
    # Then: only the non-empty name is returned
    assert result == ["ramipril"]


# Tests for _matches_any_search_term — bidirectional substring matching on a single value.


def test_matches_any_search_term_exact_match() -> None:
    assert _matches_any_search_term("aspirin", ["aspirin", "ibuprofen"]) is True


def test_matches_any_search_term_term_is_substring_of_value() -> None:
    # "acebutolol" is a substring of "acebutolol hydrochloride"
    assert _matches_any_search_term("acebutolol hydrochloride", ["acebutolol"]) is True


def test_matches_any_search_term_value_is_substring_of_term() -> None:
    # value "drug" appears inside term "new drug formulation"
    assert _matches_any_search_term("drug", ["new drug formulation"]) is True


def test_matches_any_search_term_returns_false_when_no_match() -> None:
    assert _matches_any_search_term("metformin", ["aspirin", "ibuprofen"]) is False


def test_matches_any_search_term_returns_false_for_empty_terms() -> None:
    assert _matches_any_search_term("aspirin", []) is False


# Tests for _any_substring_match — same logic, but over a set of search terms vs a list of FDA values.


def test_any_substring_match_term_in_fda_value() -> None:
    assert _any_substring_match({"acebutolol"}, ["acebutolol hydrochloride"]) is True


def test_any_substring_match_fda_value_in_term() -> None:
    assert _any_substring_match({"acebutolol hydrochloride"}, ["acebutolol"]) is True


def test_any_substring_match_returns_false_when_no_match() -> None:
    assert _any_substring_match({"metformin"}, ["aspirin", "ibuprofen"]) is False


def test_any_substring_match_returns_false_for_empty_inputs() -> None:
    assert _any_substring_match(set(), ["aspirin"]) is False
    assert _any_substring_match({"aspirin"}, []) is False


def test_any_substring_match_and_matches_any_search_term_are_equivalent() -> None:
    # Both functions implement the same bidirectional substring logic.
    # This test documents that equivalence so a future refactor doesn't silently break it.
    value = "acebutolol hydrochloride"
    terms = ["acebutolol", "metformin"]
    assert _matches_any_search_term(value, terms) == _any_substring_match({value}, terms)


# Tests for _is_anda_or_bla_application_number


def test_is_anda_or_bla_application_number_matches_anda() -> None:
    # Given: an ANDA-prefixed application number
    assert _is_anda_or_bla_application_number("ANDA123456") is True


def test_is_anda_or_bla_application_number_matches_bla() -> None:
    # Given: a BLA-prefixed application number
    assert _is_anda_or_bla_application_number("BLA654321") is True


def test_is_anda_or_bla_application_number_case_insensitive() -> None:
    # Given: lowercase prefixes — regex is case-insensitive
    assert _is_anda_or_bla_application_number("anda000001") is True
    assert _is_anda_or_bla_application_number("bla000001") is True


def test_is_anda_or_bla_application_number_rejects_nda() -> None:
    # Given: an NDA application number — should not match
    assert _is_anda_or_bla_application_number("NDA123456") is False


def test_is_anda_or_bla_application_number_rejects_empty_string() -> None:
    assert _is_anda_or_bla_application_number("") is False


# Tests for _is_anda_application_number


def test_is_anda_application_number_matches_anda() -> None:
    assert _is_anda_application_number("ANDA999999") is True


def test_is_anda_application_number_rejects_bla() -> None:
    # BLA is NOT an ANDA — only ANDA prefix should match
    assert _is_anda_application_number("BLA123456") is False


def test_is_anda_application_number_rejects_nda() -> None:
    assert _is_anda_application_number("NDA123456") is False


def test_is_anda_application_number_case_insensitive() -> None:
    assert _is_anda_application_number("anda000001") is True


# Tests for _as_list


def test_as_list_returns_list_unchanged() -> None:
    # Given: already a list — should pass through
    assert _as_list([1, 2, 3]) == [1, 2, 3]


def test_as_list_converts_tuple() -> None:
    assert _as_list((1, 2)) == [1, 2]


def test_as_list_converts_set() -> None:
    result = _as_list({42})
    assert isinstance(result, list)
    assert result == [42]


def test_as_list_converts_numpy_array() -> None:
    arr = np.array(["a", "b"], dtype=object)
    result = _as_list(arr)
    assert isinstance(result, list)
    assert result == ["a", "b"]


def test_as_list_wraps_scalar_from_tolist_like_object() -> None:
    class ScalarContainer:
        def tolist(self) -> str:
            return "value"

    assert _as_list(ScalarContainer()) == ["value"]


def test_as_list_returns_none_for_string() -> None:
    # Strings have a tolist-like interface via list() but should return None
    assert _as_list("hello") is None


def test_as_list_returns_none_for_dict() -> None:
    assert _as_list({"key": "value"}) is None


def test_as_list_returns_none_for_int() -> None:
    assert _as_list(42) is None


# Tests for _extract_strings_from_nested_value


def test_extract_strings_from_nested_value_plain_string() -> None:
    # Given: a plain string — should be normalized and returned
    assert _extract_strings_from_nested_value("  Aspirin  ") == ["aspirin"]


def test_extract_strings_from_nested_value_empty_string() -> None:
    # Empty strings are dropped
    assert _extract_strings_from_nested_value("") == []


def test_extract_strings_from_nested_value_list_of_strings() -> None:
    result = _extract_strings_from_nested_value(["Aspirin", "Ibuprofen"])
    assert result == ["aspirin", "ibuprofen"]


def test_extract_strings_from_nested_value_numpy_array() -> None:
    arr = np.array(["Drug A", "Drug B"], dtype=object)
    result = _extract_strings_from_nested_value(arr)
    assert result == ["drug a", "drug b"]


def test_extract_strings_from_nested_value_nested_list() -> None:
    # Given: nested list — all strings should be flattened and normalized
    result = _extract_strings_from_nested_value([["alpha"], ["beta", "gamma"]])
    assert result == ["alpha", "beta", "gamma"]


def test_extract_strings_from_nested_value_non_string_scalar() -> None:
    # Non-string scalars that aren't sequences return empty
    assert _extract_strings_from_nested_value(42) == []


# Tests for _extract_values_for_path


def test_extract_values_for_path_simple_key() -> None:
    # Given: a flat dict and a single-part path
    data = {"name": "Aspirin"}
    result = _extract_values_for_path(data, ["name"])
    assert result == ["aspirin"]


def test_extract_values_for_path_nested_key() -> None:
    # Given: a nested dict and a two-part path
    data = {"openfda": {"brand_name": ["BrandX"]}}
    result = _extract_values_for_path(data, ["openfda", "brand_name"])
    assert result == ["brandx"]


def test_extract_values_for_path_missing_key_returns_empty() -> None:
    data = {"openfda": {}}
    result = _extract_values_for_path(data, ["openfda", "brand_name"])
    assert result == []


def test_extract_values_for_path_list_of_dicts() -> None:
    # Given: list of dicts — should iterate and extract from each
    data = [{"name": "Alpha"}, {"name": "Beta"}]
    result = _extract_values_for_path(data, ["name"])
    assert result == ["alpha", "beta"]


def test_extract_values_for_path_none_data_returns_empty() -> None:
    assert _extract_values_for_path(None, ["name"]) == []


def test_extract_values_for_path_non_sequence_non_dict_returns_empty() -> None:
    assert _extract_values_for_path(42, ["name"]) == []


def test_extract_values_for_path_empty_path_extracts_scalar() -> None:
    # Empty path means extract from the data itself
    result = _extract_values_for_path("Warfarin", [])
    assert result == ["warfarin"]


# Tests for _unique_non_empty_strings


def test_unique_non_empty_strings_deduplicates() -> None:
    # Given: list with duplicates (case-sensitive deduplication on stripped value)
    result = _unique_non_empty_strings(["aspirin", "aspirin", "ibuprofen"])
    assert result == ["aspirin", "ibuprofen"]


def test_unique_non_empty_strings_drops_empty_and_blank() -> None:
    result = _unique_non_empty_strings(["aspirin", "", "   ", "ibuprofen"])
    assert result == ["aspirin", "ibuprofen"]


def test_unique_non_empty_strings_drops_non_strings() -> None:
    result = _unique_non_empty_strings(["aspirin", None, 42])  # type: ignore[list-item]
    assert result == ["aspirin"]


def test_unique_non_empty_strings_preserves_order() -> None:
    result = _unique_non_empty_strings(["c", "a", "b"])
    assert result == ["c", "a", "b"]


def test_unique_non_empty_strings_empty_input() -> None:
    assert _unique_non_empty_strings([]) == []


# Tests for _extract_openfda_field


def _make_filtered_rows(app_number: str, field_name: str, field_value: list) -> list[dict]:
    return [{"application_number": app_number, "openfda": {field_name: field_value}}]


def test_extract_openfda_field_returns_values() -> None:
    rows = _make_filtered_rows("ANDA111", "brand_name", ["Brand X", "Brand Y"])
    result = extract_openfda_field(rows, "brand_name")
    assert result == ["Brand X", "Brand Y"]


def test_extract_openfda_field_deduplicates() -> None:
    rows = _make_filtered_rows("ANDA111", "brand_name", ["Brand X", "Brand X"])
    result = extract_openfda_field(rows, "brand_name")
    assert result == ["Brand X"]


def test_extract_openfda_field_handles_string_value() -> None:
    # openfda field may occasionally be a plain string instead of a list
    rows = [{"openfda": {"generic_name": "aspirin"}}]
    result = extract_openfda_field(rows, "generic_name")
    assert result == ["aspirin"]


def test_extract_openfda_field_missing_field_returns_empty() -> None:
    rows = [{"openfda": {}}]
    result = extract_openfda_field(rows, "brand_name")
    assert result == []


def test_extract_openfda_field_missing_openfda_returns_empty() -> None:
    rows = [{"application_number": "ANDA111"}]
    result = extract_openfda_field(rows, "brand_name")
    assert result == []


def test_extract_openfda_field_empty_rows_returns_empty() -> None:
    assert extract_openfda_field([], "brand_name") == []


def test_extract_openfda_field_skips_non_dict_rows() -> None:
    rows = ["not-a-dict", {"openfda": {"brand_name": ["Brand X"]}}]
    result = extract_openfda_field(rows, "brand_name")
    assert result == ["Brand X"]


# Tests for _extract_product_marketing_status


def _make_rows_with_products(products: list[dict]) -> list[dict]:
    return [{"products": products}]


def test_extract_product_marketing_status_returns_unique_statuses() -> None:
    rows = _make_rows_with_products(
        [
            {"marketing_status": "Prescription"},
            {"marketing_status": "Over-the-counter"},
            {"marketing_status": "Prescription"},  # duplicate
        ]
    )
    result = extract_product_marketing_status(rows)
    assert result == ["Prescription", "Over-the-counter"]


def test_extract_product_marketing_status_skips_non_dict_products() -> None:
    rows = [{"products": ["not-a-dict", {"marketing_status": "Prescription"}]}]
    result = extract_product_marketing_status(rows)
    assert result == ["Prescription"]


def test_extract_product_marketing_status_empty_products_returns_empty() -> None:
    assert extract_product_marketing_status(_make_rows_with_products([])) == []


def test_extract_product_marketing_status_empty_rows_returns_empty() -> None:
    assert extract_product_marketing_status([]) == []


def test_extract_product_marketing_status_skips_non_dict_rows_and_none_products() -> None:
    rows = [
        "not-a-dict",
        {"products": None},
        {"products": [{"marketing_status": "Prescription"}]},
    ]
    result = extract_product_marketing_status(rows)
    assert result == ["Prescription"]


# Tests for _extract_product_active_ingredients


def test_extract_product_active_ingredients_from_dict_ingredients() -> None:
    rows = _make_rows_with_products([{"active_ingredients": [{"name": "aspirin"}, {"name": "caffeine"}]}])
    result = extract_product_active_ingredients(rows)
    assert result == ["aspirin", "caffeine"]


def test_extract_product_active_ingredients_from_string_ingredients() -> None:
    rows = _make_rows_with_products([{"active_ingredients": ["warfarin"]}])
    result = extract_product_active_ingredients(rows)
    assert result == ["warfarin"]


def test_extract_product_active_ingredients_deduplicates() -> None:
    rows = _make_rows_with_products(
        [
            {"active_ingredients": [{"name": "aspirin"}]},
            {"active_ingredients": [{"name": "aspirin"}]},
        ]
    )
    result = extract_product_active_ingredients(rows)
    assert result == ["aspirin"]


def test_extract_product_active_ingredients_empty_rows_returns_empty() -> None:
    assert extract_product_active_ingredients([]) == []


def test_extract_product_active_ingredients_handles_numpy_array_products() -> None:
    rows = [
        {
            "products": np.array(
                [
                    {
                        "active_ingredients": np.array(
                            [{"name": "aspirin"}, {"name": "caffeine"}],
                            dtype=object,
                        )
                    }
                ],
                dtype=object,
            )
        }
    ]
    result = extract_product_active_ingredients(rows)
    assert result == ["aspirin", "caffeine"]


def test_extract_product_active_ingredients_skips_invalid_rows_and_products() -> None:
    rows = [
        "not-a-dict",
        {"products": None},
        {
            "products": [
                "not-a-dict",
                {"active_ingredients": None},
                {"active_ingredients": [{"name": "aspirin"}]},
            ]
        },
    ]
    result = extract_product_active_ingredients(rows)
    assert result == ["aspirin"]


# Tests for _has_anda_application_number


def test_has_anda_application_number_returns_true_when_anda_present() -> None:
    rows = [{"application_number": "ANDA123456"}]
    assert has_anda_application_number(rows) is True


def test_has_anda_application_number_returns_false_for_bla_only() -> None:
    rows = [{"application_number": "BLA654321"}]
    assert has_anda_application_number(rows) is False


def test_has_anda_application_number_returns_false_for_nda_only() -> None:
    rows = [{"application_number": "NDA111111"}]
    assert has_anda_application_number(rows) is False


def test_has_anda_application_number_returns_true_when_mixed() -> None:
    # At least one ANDA row among others is enough
    rows = [{"application_number": "NDA111111"}, {"application_number": "ANDA222222"}]
    assert has_anda_application_number(rows) is True


def test_has_anda_application_number_empty_rows_returns_false() -> None:
    assert has_anda_application_number([]) is False


def test_has_anda_application_number_skips_non_dict_rows() -> None:
    assert has_anda_application_number(["not-a-dict"]) is False  # type: ignore[list-item]


# Tests for _active_ingredients_match_combo


def test_active_ingredients_match_combo_valid_pair() -> None:
    # Given: 2-ingredient product matching one search term and one combo partner
    ingredients = [{"name": "drug a"}, {"name": "drug b"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is True


def test_active_ingredients_match_combo_rejects_single_ingredient() -> None:
    # Must have exactly 2 ingredients
    ingredients = [{"name": "drug a"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is False


def test_active_ingredients_match_combo_rejects_three_ingredients() -> None:
    ingredients = [{"name": "drug a"}, {"name": "drug b"}, {"name": "drug c"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is False


def test_active_ingredients_match_combo_requires_drug_match() -> None:
    # Neither ingredient matches the search terms
    ingredients = [{"name": "drug x"}, {"name": "drug b"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is False


def test_active_ingredients_match_combo_requires_partner_match() -> None:
    # Drug matches but combo partner doesn't
    ingredients = [{"name": "drug a"}, {"name": "drug x"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is False


def test_active_ingredients_match_combo_is_bidirectional_substring() -> None:
    # "drug a hydrochloride" contains "drug a" — substring match should work
    ingredients = [{"name": "drug a hydrochloride"}, {"name": "drug b"}]
    assert _active_ingredients_match_combo(ingredients, ["drug a"], "drug b") is True


# Tests for match_drug_to_fda_worker


def test_match_drug_to_fda_worker_returns_expected_keys() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "ANDA111111",
                "openfda": {"brand_name": ["Brand A"], "generic_name": ["alpha"]},
                "products": [{"active_ingredients": [{"name": "alpha"}]}],
            }
        ]
    )
    work_item = {
        "name": "alpha drug",
        "id": "EC:00001",
        "search_terms": {"alpha"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["openfda.generic_name"],
    )

    assert "fda_rows" in result
    assert "fda_match_count" in result
    assert "drug_name" in result
    assert "id" in result
    assert "search_terms" in result
    assert "available_in_combo_with" in result

    assert result == {
        "fda_rows": [
            {
                "application_number": "ANDA111111",
                "openfda": {"brand_name": ["Brand A"], "generic_name": ["alpha"]},
                "products": [{"active_ingredients": [{"name": "alpha"}]}],
            }
        ],
        "fda_match_count": 1,
        "drug_name": "alpha drug",
        "id": "EC:00001",
        "search_terms": ["alpha"],
        "available_in_combo_with": [],
    }


def test_match_drug_to_fda_worker_finds_matching_row() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "ANDA111111",
                "openfda": {"generic_name": ["metformin"]},
                "products": [],
            },
            {
                "application_number": "ANDA222222",
                "openfda": {"generic_name": ["warfarin"]},
                "products": [],
            },
        ]
    )
    work_item = {
        "name": "metformin drug",
        "id": "EC:00002",
        "search_terms": {"metformin"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["openfda.generic_name"],
    )

    assert result["fda_match_count"] == 1
    assert result["fda_rows"][0]["application_number"] == "ANDA111111"
    assert result["drug_name"] == "metformin drug"
    assert result["id"] == "EC:00002"
    assert result["search_terms"] == ["metformin"]


def test_match_drug_to_fda_worker_no_match_returns_empty_rows() -> None:
    fda_drug_list = pd.DataFrame(
        [{"application_number": "ANDA111111", "openfda": {"generic_name": ["warfarin"]}, "products": []}]
    )
    work_item = {
        "name": "unknown drug",
        "id": "EC:00003",
        "search_terms": {"nosuchdrug"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["openfda.generic_name"],
    )

    assert result["fda_match_count"] == 0
    assert result["fda_rows"] == []
