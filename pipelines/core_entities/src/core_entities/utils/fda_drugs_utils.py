import logging
import re
from typing import Any, Iterator, Sequence

import pandas as pd

from core_entities.utils.python_utils import canonicalize_reference_flags, deep_lowercase_strings

logger = logging.getLogger(__name__)


_COMBO_KEYWORD_PATTERN = re.compile(r"\band\b|&|;|,", re.IGNORECASE)
_COMBO_SPLIT_PATTERN = re.compile(r"\s+\band\b\s+|\s*&\s*|\s*;\s*|\s*,\s*", re.IGNORECASE)


def _normalize(value: str) -> str:
    """Lowercase and strip whitespace — single source of truth for string normalisation."""
    return value.lower().strip()


def normalize_fda_results_to_dataframe(fda_drug_list: dict) -> pd.DataFrame:
    normalized_results = deep_lowercase_strings(fda_drug_list["results"])
    normalized_results = canonicalize_reference_flags(normalized_results)
    return pd.DataFrame(normalized_results)


def _has_application_prefix(application_number: str, *prefixes: str) -> bool:
    """Check whether an application number starts with any of the given prefixes."""
    normalized_application_number = _normalize(str(application_number))
    normalized_prefixes = tuple(
        _normalize(prefix) for prefix in prefixes if isinstance(prefix, str) and _normalize(prefix) != ""
    )
    if not normalized_prefixes:
        return False
    return normalized_application_number.startswith(normalized_prefixes)


def _as_list(value: Any) -> list | None:
    """Coerce sequences (list, tuple, set, ndarray) to list; return None for non-sequences."""
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (dict, str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return None


def _extract_strings_from_nested_value(value: Any) -> list[str]:
    """Recursively extract, normalise, and flatten strings from arbitrarily nested data."""
    if isinstance(value, str):
        normalized = _normalize(value)
        return [normalized] if normalized else []

    sequence_value = _as_list(value)
    if sequence_value is not None:
        values: list[str] = []
        for item in sequence_value:
            values.extend(_extract_strings_from_nested_value(item))
        return values

    return []


def _extract_values_for_path(data: Any, path_parts: list[str]) -> list[str]:
    """Extract normalised strings from nested data for a dotted path."""
    if data is None:
        return []

    if not path_parts:
        return _extract_strings_from_nested_value(data)

    current_key = path_parts[0]
    remaining_path = path_parts[1:]

    if isinstance(data, dict):
        if current_key not in data:
            return []
        return _extract_values_for_path(data[current_key], remaining_path)

    sequence_data = _as_list(data)
    if sequence_data is not None:
        values: list[str] = []
        for item in sequence_data:
            values.extend(_extract_values_for_path(item, path_parts))
        return values

    return []


def _find_matching_fda_rows(
    search_terms: set[str],
    fda_drug_list: pd.DataFrame,
    fda_drug_list_columns_to_use_for_matching: list[str],
) -> list[dict]:
    """Find FDA rows where configured fields match any of the search terms.

    Uses substring matching so that e.g. search term "acebutolol" matches
     the FDA field "acebutolol hydrochloride" and vice versa.
    """
    normalized_search_terms = {
        _normalize(term) for term in search_terms if isinstance(term, str) and term.strip() != ""
    }

    # Pre-filter column paths once rather than per-row.
    valid_column_paths = [
        column_path.strip()
        for column_path in (fda_drug_list_columns_to_use_for_matching or [])
        if isinstance(column_path, str) and column_path.strip() != ""
    ]

    if not normalized_search_terms or not valid_column_paths:
        return []

    matching_rows = []
    for row_dict in fda_drug_list.to_dict("records"):
        fda_field_values: list[str] = []
        for column_path in valid_column_paths:
            fda_field_values.extend(_extract_values_for_path(row_dict, column_path.split(".")))
        fda_field_values = _unique_non_empty_strings(fda_field_values)

        if _any_substring_match(normalized_search_terms, fda_field_values):
            matching_rows.append(row_dict)

    return matching_rows


def _any_substring_match(search_terms: Sequence[str], fda_field_values: Sequence[str]) -> bool:
    """Check if any search term is a substring of any FDA field value or vice versa."""
    for term in search_terms:
        for fda_value in fda_field_values:
            if term in fda_value or fda_value in term:
                return True
    return False


def _normalize_string_terms(values: str | Sequence[Any] | None) -> list[str]:
    """Flatten *values* into a list of lowered/stripped non-empty strings.

    Delegates to `_extract_strings_from_nested_value` to avoid duplicating
    the normalise-and-flatten logic.
    """
    if values is None:
        return []
    return _extract_strings_from_nested_value(values)


def _split_combo_expression_parts(value: str) -> list[str]:
    """Split strings on supported combo separators while preserving normalised tokens."""
    normalized = _normalize(value)
    if normalized == "":
        return []
    return [part.strip() for part in _COMBO_SPLIT_PATTERN.split(normalized) if part.strip() != ""]


def _is_likely_drug_like_combo_part(value: str) -> bool:
    """Conservative shape check to avoid excluding clearly invalid split fragments."""
    normalized = value.strip()
    if len(normalized) < 3:
        return False
    return bool(re.search(r"[a-z]", normalized))


def _is_probable_combo_expression(value: str, search_terms: list[str]) -> bool:
    """Return True for likely combo strings (e.g. 'drug a and drug b', 'drug a & drug b', 'drug a; drug b')."""
    normalized = _normalize(value)
    if normalized == "":
        return False

    # If the full value is itself an accepted search term, keep it as a legitimate single ingredient label.
    search_terms_set = set(search_terms)
    if normalized in search_terms_set:
        return False

    if not _COMBO_KEYWORD_PATTERN.search(normalized):
        return False

    parts = _split_combo_expression_parts(normalized)
    if len(parts) < 2:
        return False

    if len(set(parts)) < 2:
        return False

    if not all(_is_likely_drug_like_combo_part(part) for part in parts):
        return False

    # Require one part that matches the current drug context and another that does not.
    has_matching_part = any(_any_substring_match((part,), search_terms) for part in parts)
    has_non_matching_part = any(not _any_substring_match((part,), search_terms) for part in parts)
    return has_matching_part and has_non_matching_part


def _has_probable_combo_substance_name(fda_row: dict, search_terms: list[str]) -> bool:
    """Check row-level openfda.substance_name values for likely combo strings."""
    substance_names = _extract_values_for_path(fda_row, ["openfda", "substance_name"])
    return any(_is_probable_combo_expression(name, search_terms) for name in substance_names)


def _extract_active_ingredient_names(active_ingredients: list[dict | str]) -> list[str]:
    """Pull the name from each ingredient (dict with 'name' or bare string) and normalise."""
    raw_names: list[Any] = [
        ingredient.get("name") if isinstance(ingredient, dict) else ingredient
        for ingredient in active_ingredients
        if isinstance(ingredient, (dict, str))
    ]
    return _normalize_string_terms(raw_names)


def match_drug_to_fda_worker(
    work_item: dict,
    fda_drug_list: pd.DataFrame,
    fda_drug_list_columns_to_use_for_matching: list[str],
) -> dict:
    """Worker function for multiprocessing: match a single drug to FDA rows."""
    search_terms = work_item["search_terms"]
    logger.debug("Matching drug %s with %d search terms", work_item["name"], len(search_terms))

    matching_fda_rows = _find_matching_fda_rows(
        search_terms=search_terms,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=fda_drug_list_columns_to_use_for_matching,
    )

    return {
        "fda_rows": matching_fda_rows,
        "fda_match_count": len(matching_fda_rows),
        "drug_name": work_item["name"],
        "id": work_item["id"],
        "search_terms": sorted(search_terms),
        "available_in_combo_with": work_item["available_in_combo_with"],
    }


def _active_ingredients_match_combo(
    active_ingredients: list[dict],
    search_terms: list[str],
    combo_partner: str,  # one partner at a time — expected to be pre-normalised
) -> bool:
    """Check if a product's active ingredients contain:
    - exactly 2 ingredients
    - one that matches any of the search_terms (the drug itself)
    - one that matches the combo_partner
    """
    ingredient_names = _extract_active_ingredient_names(active_ingredients)
    if len(ingredient_names) != 2:
        return False

    # Find the ingredient that matches the drug (any search term)
    drug_match_idx = next(
        (i for i, name in enumerate(ingredient_names) if _any_substring_match((name,), search_terms)), None
    )
    if drug_match_idx is None:
        return False

    # Find a DIFFERENT ingredient that matches the combo partner
    combo_match_idx = next(
        (
            i
            for i, name in enumerate(ingredient_names)
            if i != drug_match_idx and _any_substring_match((name,), [combo_partner])
        ),
        None,
    )

    return combo_match_idx is not None


def _is_active_marketing_status(product: dict) -> bool:
    """Check whether a product has an active marketing status (prescription or OTC)."""
    return _normalize(str(product.get("marketing_status", ""))) in ("prescription", "over-the-counter")


def _filter_matching_products(
    products: list,
    search_terms: list[str],
    combo_partners: list[str],
    has_combo_substance_name: bool,
) -> list[dict]:
    """Return product dicts that match the drug based on active ingredients and marketing status."""
    matching: list[dict] = []

    for product in products:
        if not isinstance(product, dict):
            continue
        if not _is_active_marketing_status(product):
            continue

        active_ingredients = _as_list(product.get("active_ingredients"))
        if active_ingredients is None:
            continue

        # Combo path: check if a product pairs the drug with any known combo partner.
        if combo_partners:
            if any(
                _active_ingredients_match_combo(active_ingredients, search_terms, partner) for partner in combo_partners
            ):
                matching.append(product)
            continue

        # Single-ingredient path.
        ingredient_names = _extract_active_ingredient_names(active_ingredients)
        if len(ingredient_names) != 1:
            continue

        ingredient_name = ingredient_names[0]
        if _is_probable_combo_expression(ingredient_name, search_terms):
            continue
        if has_combo_substance_name:
            continue
        if _any_substring_match((ingredient_name,), search_terms):
            matching.append(product)

    return matching


def filter_fda_rows(row: pd.Series) -> list:
    """Filter FDA rows to only include matching products for a given drug."""
    matches: list[dict] = []
    normalized_search_terms = _normalize_string_terms(row.get("search_terms", []))
    normalized_combo_partners = _normalize_string_terms(row.get("available_in_combo_with", []))

    fda_rows = _as_list(row.get("fda_rows"))
    if fda_rows is None:
        return matches

    for value in fda_rows:
        if not isinstance(value, dict):
            continue
        if not _has_application_prefix(value.get("application_number", ""), "anda", "bla"):
            continue

        has_combo_substance_name = _has_probable_combo_substance_name(value, normalized_search_terms)

        products = _as_list(value.get("products"))
        if products is None:
            continue

        matching_products = _filter_matching_products(
            products, normalized_search_terms, normalized_combo_partners, has_combo_substance_name
        )
        if matching_products:
            value_match = dict(value)
            value_match["products"] = matching_products
            matches.append(value_match)

    return matches


def _unique_non_empty_strings(values: list) -> list[str]:
    """Deduplicate and strip strings, preserving insertion order and dropping blanks/non-strings."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped == "" or stripped in seen:
            continue
        seen.add(stripped)
        output.append(stripped)
    return output


def extract_openfda_field(filtered_rows: list, field_name: str) -> list[str]:
    values = []
    for row in filtered_rows or []:
        if not isinstance(row, dict):
            continue
        openfda = row.get("openfda")
        if not isinstance(openfda, dict):
            continue

        field_value = openfda.get(field_name, [])
        if isinstance(field_value, str):
            values.append(field_value)
        elif isinstance(field_value, list):
            values.extend(field_value)

    return _unique_non_empty_strings(values)


def _iter_product_dicts(filtered_rows: list) -> Iterator[dict]:
    """Yield product dicts from filtered FDA rows, skipping malformed items."""
    for row in filtered_rows or []:
        if not isinstance(row, dict):
            continue
        products = _as_list(row.get("products", []))
        if products is None:
            continue
        for product in products:
            if isinstance(product, dict):
                yield product


def extract_product_marketing_status(filtered_rows: list) -> list[str]:
    values = []
    for product in _iter_product_dicts(filtered_rows):
        values.append(product.get("marketing_status"))

    return _unique_non_empty_strings(values)


def extract_product_active_ingredients(filtered_rows: list) -> list[str]:
    values = []
    for product in _iter_product_dicts(filtered_rows):
        active_ingredients = _as_list(product.get("active_ingredients", []))
        if active_ingredients is None:
            continue
        for ingredient in active_ingredients:
            if isinstance(ingredient, dict):
                values.append(ingredient.get("name"))
            elif isinstance(ingredient, str):
                values.append(ingredient)

    return _unique_non_empty_strings(values)


def has_anda_application_number(fda_rows: list) -> bool:
    for row in fda_rows or []:
        if not isinstance(row, dict):
            continue
        application_number = row.get("application_number")
        if _has_application_prefix(application_number, "anda"):
            return True
    return False
