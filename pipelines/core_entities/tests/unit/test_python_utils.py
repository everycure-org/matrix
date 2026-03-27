import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.utils.python_utils import (
    _deep_to_python,
    canonicalize_reference_flags,
    deep_lowercase_strings,
    ensure_python_list,
    ensure_string_list,
)


def test_deep_to_python_converts_numpy_scalar_to_python_scalar() -> None:
    result = _deep_to_python(np.int64(7))
    assert result == 7
    assert isinstance(result, int)


def test_deep_to_python_recursively_converts_nested_dict_and_array() -> None:
    payload = {
        "count": np.int64(3),
        "items": np.array([np.float64(1.5), {"flag": np.bool_(True)}], dtype=object),
    }

    result = _deep_to_python(payload)

    assert result == {"count": 3, "items": [1.5, {"flag": True}]}


def test_deep_to_python_passthrough_for_plain_object() -> None:
    marker = object()
    assert _deep_to_python(marker) is marker


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
        "reference_drug": "yes",
        "reference_standard": "no",
        "other": "Y",
    }


def test_canonicalize_reference_flags_handles_bool_and_nullish_values() -> None:
    payload = {
        "reference_drug": True,
        "reference_standard": "N/A",
    }

    result = canonicalize_reference_flags(payload)

    assert result == {
        "reference_drug": "yes",
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
            "reference_drug": "yes",
            "nested": {
                "reference_standard": "no",
                "note": "KeepMe",
            },
        },
        {
            "reference_drug": "no",
            "reference_standard": None,
        },
    ]


def test_canonicalize_reference_flags_returns_scalar_unchanged() -> None:
    assert canonicalize_reference_flags("value") == "value"
    assert canonicalize_reference_flags(99) == 99


def test_ensure_string_list_returns_empty_for_pandas_na_scalar() -> None:
    assert ensure_string_list(pd.NA) == []
