import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.utils.python_utils import (
    canonicalize_reference_flags,
    deep_lowercase_strings,
    ensure_python_list,
    ensure_string_list,
    merge_unique_strings,
    normalize_bla_number,
    resolve_column_name,
)


def test_ensure_python_list_converts_list_values_deeply() -> None:
    result = ensure_python_list([np.int64(1), {"x": np.bool_(False)}])
    assert result == [1, {"x": False}]


def test_ensure_python_list_converts_numpy_array_to_list() -> None:
    arr = np.array(["a", np.int64(2)], dtype=object)
    assert ensure_python_list(arr) == ["a", 2]


def test_ensure_python_list_returns_original_for_non_sequence() -> None:
    assert ensure_python_list("value") == "value"
    assert ensure_python_list(42) == 42


def test_ensure_string_list_returns_empty_for_none() -> None:
    assert ensure_string_list(None) == []


def test_ensure_string_list_returns_empty_for_nan_scalar() -> None:
    assert ensure_string_list(np.nan) == []


def test_ensure_string_list_wraps_string() -> None:
    assert ensure_string_list("alpha") == ["alpha"]


def test_ensure_string_list_filters_non_strings_from_list_like() -> None:
    assert ensure_string_list(["alpha", 1, None, "beta"]) == ["alpha", "beta"]
    assert ensure_string_list(("x", 2)) == ["x"]
    # Set ordering is nondeterministic, so sort for assertion.
    assert sorted(ensure_string_list({"x", 2})) == ["x"]


def test_ensure_string_list_handles_numpy_array_input() -> None:
    arr = np.array(["alpha", 1, "beta"], dtype=object)
    assert ensure_string_list(arr) == ["alpha", "beta"]


def test_ensure_string_list_returns_empty_for_other_scalars() -> None:
    assert ensure_string_list(123) == []


def test_deep_lowercase_strings_normalizes_nested_structures() -> None:
    payload = {
        "name": "  ASPIRIN ",
        "aliases": [" Acetylsalicylic Acid ", "ASA"],
        "meta": np.array([{"route": " ORAL "}], dtype=object),
        "count": np.int64(2),
    }

    result = deep_lowercase_strings(payload)

    assert result == {
        "name": "aspirin",
        "aliases": ["acetylsalicylic acid", "asa"],
        "meta": [{"route": "oral"}],
        "count": 2,
    }


def test_canonicalize_reference_flags_normalizes_flag_fields() -> None:
    payload = {
        "reference_drug": "Y",
        "reference_standard": " false ",
        "other": "Y",
    }

    result = canonicalize_reference_flags(payload)

    assert result == {
        "reference_drug": True,
        "reference_standard": False,
        "other": "Y",
    }


def test_canonicalize_reference_flags_handles_bool_and_nullish_values() -> None:
    payload = {
        "reference_drug": True,
        "reference_standard": "N/A",
    }

    result = canonicalize_reference_flags(payload)

    assert result == {
        "reference_drug": True,
        "reference_standard": None,
    }


def test_canonicalize_reference_flags_leaves_unknown_flag_values_unchanged() -> None:
    payload = {
        "reference_drug": "maybe",
        "reference_standard": 123,
    }

    result = canonicalize_reference_flags(payload)

    assert result == {
        "reference_drug": "maybe",
        "reference_standard": 123,
    }


def test_canonicalize_reference_flags_recurses_into_lists_and_nested_dicts() -> None:
    payload = [
        {
            "reference_drug": "1",
            "nested": {
                "reference_standard": "0",
                "note": "KeepMe",
            },
        },
        {
            "reference_drug": np.bool_(False),
            "reference_standard": "unknown",
        },
    ]

    result = canonicalize_reference_flags(payload)

    assert result == [
        {
            "reference_drug": True,
            "nested": {
                "reference_standard": False,
                "note": "KeepMe",
            },
        },
        {
            "reference_drug": False,
            "reference_standard": None,
        },
    ]


def test_canonicalize_reference_flags_returns_scalar_unchanged() -> None:
    assert canonicalize_reference_flags("value") == "value"
    assert canonicalize_reference_flags(99) == 99


def test_ensure_string_list_returns_empty_for_pandas_na_scalar() -> None:
    assert ensure_string_list(pd.NA) == []


def test_resolve_column_name_matches_case_insensitive_trimmed_names() -> None:
    df = pd.DataFrame(columns=[" BLA Number ", "BLA Type", "Other"])
    result = resolve_column_name(df, ["bla number", "bla_number"])
    assert result == " BLA Number "


def test_resolve_column_name_returns_none_when_no_candidate_matches() -> None:
    df = pd.DataFrame(columns=["foo", "bar"])
    assert resolve_column_name(df, ["baz", "qux"]) is None


def test_normalize_bla_number_extracts_digits_and_removes_leading_zeros() -> None:
    assert normalize_bla_number(" BLA 000761071 ") == "761071"
    assert normalize_bla_number(761071) == "761071"


def test_normalize_bla_number_returns_none_for_empty_or_non_digit_values() -> None:
    assert normalize_bla_number("") is None
    assert normalize_bla_number("   ") is None
    assert normalize_bla_number("BLA") is None


def test_merge_unique_strings_deduplicates_case_insensitive_and_preserves_first_seen() -> None:
    values = [["A", " a ", "B"], "b", ["C", "c"], None, [1, "D"]]
    assert merge_unique_strings(values) == ["A", "B", "C", "D"]


def test_merge_unique_strings_supports_mixed_list_like_values() -> None:
    values = [("x", "Y", 1), {"y", "Z"}, np.array(["z", "W"], dtype=object)]
    result = merge_unique_strings(values)
    assert set(result) == {"x", "Y", "Z", "W"}
    assert len(result) == 4
