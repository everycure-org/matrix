import numpy as np
import pandas as pd


def _deep_to_python(obj):
    """Recursively convert numpy/arrow types to plain Python objects."""
    if isinstance(obj, dict):
        return {k: _deep_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, np.ndarray)):
        return [_deep_to_python(i) for i in obj]
    elif isinstance(obj, np.generic):  # numpy scalars e.g. np.int64, np.bool_
        return obj.item()
    return obj


def ensure_python_list(val):
    if isinstance(val, (list, np.ndarray)):
        return [_deep_to_python(i) for i in val]
    return val


def ensure_string_list(value):
    value = ensure_python_list(value)

    if value is None or (np.isscalar(value) and pd.isna(value)):
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str)]

    return []


def deep_lowercase_strings(obj):
    """Recursively lowercase and strip all string values in nested objects."""

    def _lowercase(value):
        if isinstance(value, dict):
            return {k: _lowercase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_lowercase(item) for item in value]
        if isinstance(value, str):
            return value.strip().lower()
        return value

    return _lowercase(_deep_to_python(obj))


_REFERENCE_FLAG_FIELDS = {"reference_drug", "reference_standard"}
_NULLISH_REFERENCE_VALUES = {"", "none", "null", "na", "n/a", "unknown", "tbd"}


def canonicalize_reference_flags(obj):
    """Normalise reference flags to canonical True/False/None values."""

    def _normalize_flag(value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "y", "true", "1"}:
                return True
            if normalized in {"no", "n", "false", "0"}:
                return False
            if normalized in _NULLISH_REFERENCE_VALUES:
                return None

        return value

    def _walk(value):
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key in _REFERENCE_FLAG_FIELDS:
                    output[key] = _normalize_flag(item)
                else:
                    output[key] = _walk(item)
            return output

        if isinstance(value, list):
            return [_walk(item) for item in value]

        return value

    return _walk(_deep_to_python(obj))
