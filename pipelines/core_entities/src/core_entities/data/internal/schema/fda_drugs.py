"""Pandera schema definitions for the ingested FDA drugs DataFrame.

The schema reflects the structure returned by ``ingest_fda_drug_list``, which
flattens the ``results`` array from the openFDA drug-applications JSON feed:
https://open.fda.gov/apis/drug/drugsfda/

Each row represents one NDA/ANDA/BLA application and contains nested data
for its submissions, openfda metadata, and product list.

All string values are expected to be lowercase before validation runs.
"""

import re

import pandera.pandas as pa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_APPLICATION_NUMBER_PATTERN = re.compile(r"^(nda|anda|bla)\d+$")

# Allowed values for products[*].marketing_status (lowercase).
_VALID_MARKETING_STATUSES = frozenset(
    {
        "prescription",
        "over-the-counter",
        "discontinued",
        "none (tentative approval)",
    }
)

# Known string sub-keys inside the openfda dict whose values are lists of strings.
_OPENFDA_STRING_LIST_KEYS = frozenset(
    {
        "brand_name",
        "generic_name",
        "substance_name",
        "manufacturer_name",
        "product_ndc",
        "route",
        "pharm_class_epc",
        "pharm_class_moa",
        "pharm_class_cs",
        "pharm_class_pe",
        "application_number",
        "package_ndc",
        "unii",
        "rxcui",
        "spl_id",
        "spl_set_id",
        "product_type",
    }
)

# ---------------------------------------------------------------------------
# Helpers — openfda
# ---------------------------------------------------------------------------


def _openfda_is_dict(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell or a plain dict."""
    return series.apply(lambda x: isinstance(x, dict))


def _openfda_string_lists_are_lowercase(series: pa.typing.Series) -> pa.typing.Series:
    """All string values in the known openfda list-of-strings fields must be lowercase."""

    def _check_entry(openfda: dict) -> bool:
        for key in _OPENFDA_STRING_LIST_KEYS:
            values = openfda.get(key, [])
            if not isinstance(values, list):
                continue
            for v in values:
                if isinstance(v, str) and v != v.lower():
                    return False
        return True

    return series.apply(_check_entry)


def _openfda_string_lists_are_non_empty(series: pa.typing.Series) -> pa.typing.Series:
    """Known openfda list fields must not contain blank strings."""

    def _check_entry(openfda: dict) -> bool:
        for key in _OPENFDA_STRING_LIST_KEYS:
            values = openfda.get(key, [])
            if not isinstance(values, list):
                continue
            for v in values:
                if isinstance(v, str) and v.strip() == "":
                    return False
        return True

    return series.apply(_check_entry)


# ---------------------------------------------------------------------------
# Helpers — products
# ---------------------------------------------------------------------------

_VALID_REFERENCE_FLAG = frozenset({"yes", "no"})


def _products_are_list(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell or a list of dicts."""
    return series.apply(lambda x: isinstance(x, list) and all(isinstance(p, dict) for p in x))


def _products_marketing_status_valid(series: pa.typing.Series) -> pa.typing.Series:
    """Every product's marketing_status must be one of the known lowercase values."""

    def _check_entry(products: list) -> bool:
        for product in products:
            if not isinstance(product, dict):
                continue
            status = product.get("marketing_status")
            if status is not None and status not in _VALID_MARKETING_STATUSES:
                return False
        return True

    return series.apply(_check_entry)


def _products_active_ingredients_lowercase(series: pa.typing.Series) -> pa.typing.Series:
    """active_ingredients[*].name and .strength must be lowercase strings."""

    def _check_entry(products: list) -> bool:
        for product in products:
            if not isinstance(product, dict):
                continue
            for ingredient in product.get("active_ingredients", []) or []:
                if not isinstance(ingredient, dict):
                    continue
                for field in ("name", "strength"):
                    v = ingredient.get(field)
                    if isinstance(v, str) and v != v.lower():
                        return False
        return True

    return series.apply(_check_entry)


def _products_string_fields_lowercase(series: pa.typing.Series) -> pa.typing.Series:
    """Top-level string fields on each product dict must be lowercase."""
    _string_fields = ("brand_name", "dosage_form", "route", "marketing_status", "te_code")

    def _check_entry(products: list) -> bool:
        for product in products:
            if not isinstance(product, dict):
                continue
            for field in _string_fields:
                v = product.get(field)
                if isinstance(v, str) and v != v.lower():
                    return False
        return True

    return series.apply(_check_entry)


def _products_reference_flags_valid(series: pa.typing.Series) -> pa.typing.Series:
    """reference_drug and reference_standard flags must be 'yes' or 'no' (lowercase)."""

    def _check_entry(products: list) -> bool:
        for product in products:
            if not isinstance(product, dict):
                continue
            for field in ("reference_drug", "reference_standard"):
                v = product.get(field)
                if v is not None and v not in _VALID_REFERENCE_FLAG:
                    return False
        return True

    return series.apply(_check_entry)


# ---------------------------------------------------------------------------
# Helpers — submissions
# ---------------------------------------------------------------------------


def _submissions_are_list(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell must be list of submission dicts."""
    return series.apply(lambda x: isinstance(x, list) and all(isinstance(s, dict) for s in x))


def _list_like_of_dicts(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell or a list-like collection of dicts."""

    def _check_entry(value: object) -> bool:
        if isinstance(value, list):
            return all(isinstance(item, dict) for item in value)
        if isinstance(value, (tuple, set)):
            return all(isinstance(item, dict) for item in list(value))
        if hasattr(value, "tolist") and not isinstance(value, (dict, str, bytes)):
            converted = value.tolist()
            if not isinstance(converted, list):
                return False
            return all(isinstance(item, dict) for item in converted)
        return False

    return series.apply(_check_entry)


# ---------------------------------------------------------------------------
# Schema
# (output of ``ingest_fda_drug_list``)
# ---------------------------------------------------------------------------

FDA_DRUG_LIST_SCHEMA = pa.DataFrameSchema(
    columns={
        # Primary key — nda/anda/bla number (lowercase), e.g. "nda021274"
        "application_number": pa.Column(
            dtype=str,
            nullable=False,
            checks=[
                pa.Check(
                    lambda col: col.apply(
                        lambda x: bool(_VALID_APPLICATION_NUMBER_PATTERN.match(str(x))) if isinstance(x, str) else False
                    ),
                    title="application_number must start with nda, anda, or bla followed by digits (lowercase)",
                ),
                pa.Check(
                    lambda col: col.is_unique,
                    title="application_number must be unique across rows",
                ),
                pa.Check(
                    lambda col: col.apply(lambda x: x == x.lower()),
                    title="application_number must be lowercase",
                ),
            ],
        ),
        # Sponsor/company name (lowercase)
        "sponsor_name": pa.Column(
            dtype=str,
            nullable=False,
            checks=[
                pa.Check(
                    lambda col: col.apply(lambda x: isinstance(x, str) and x.strip() != ""),
                    title="sponsor_name must be a non-empty string",
                ),
                pa.Check(
                    lambda col: col.apply(lambda x: x == x.lower()),
                    title="sponsor_name must be lowercase",
                ),
            ],
        ),
        # List of regulatory submission dicts (type, number, status, etc.).
        # May be absent for some applications.
        "submissions": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _submissions_are_list,
                    title="submissions must be a list of dicts",
                ),
            ],
        ),
        # openFDA-enriched metadata dict. Sub-keys (brand_name, generic_name,
        # substance_name, manufacturer_name, route, etc.) hold lists of strings.
        # All string values must be lowercase.
        "openfda": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _openfda_is_dict,
                    title="openfda or a dict",
                ),
                pa.Check(
                    _openfda_string_lists_are_lowercase,
                    title="openfda string list values must be lowercase",
                ),
                pa.Check(
                    _openfda_string_lists_are_non_empty,
                    title="openfda string list values must not be blank",
                ),
            ],
        ),
        # List of product dicts. Each product must have:
        #   - product_number (str)
        #   - reference_drug (str: "yes"/"no", lowercase)
        #   - brand_name (str, lowercase)
        #   - active_ingredients (list of {"name": str, "strength": str}, lowercase)
        #   - reference_standard (str: "yes"/"no", lowercase)
        #   - dosage_form (str, lowercase)
        #   - route (str, lowercase)
        #   - marketing_status (str: one of _VALID_MARKETING_STATUSES, lowercase)
        "products": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _products_are_list,
                    title="products must be a list of dicts",
                ),
                pa.Check(
                    _products_marketing_status_valid,
                    title=f"products[*].marketing_status must be one of {sorted(_VALID_MARKETING_STATUSES)}",
                ),
                pa.Check(
                    _products_string_fields_lowercase,
                    title="products[*] string fields (brand_name, dosage_form, route, marketing_status, te_code) must be lowercase",
                ),
                pa.Check(
                    _products_active_ingredients_lowercase,
                    title="products[*].active_ingredients[*].name and .strength must be lowercase",
                ),
                pa.Check(
                    _products_reference_flags_valid,
                    title="products[*].reference_drug and .reference_standard must be 'yes' or 'no' (lowercase)",
                ),
            ],
        ),
    },
    # Enforce that *only* these five columns are present after ingestion.
    strict=True,
    unique=["application_number"],
    # A completely empty result set is not expected from a healthy feed.
    # Remove this check if empty DataFrames are a valid state in your pipeline.
    checks=[
        pa.Check(
            lambda df: len(df) > 0,
            title="FDA drug list must not be empty",
        ),
    ],
)


FDA_DRUG_LIST_FOR_MATCHING_SCHEMA = pa.DataFrameSchema(
    columns={
        "application_number": pa.Column(nullable=False),
        "submissions": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _list_like_of_dicts,
                    title="submissions or a list-like collection of dicts",
                ),
            ],
        ),
        "openfda": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _openfda_is_dict,
                    title="openfda or a dict",
                ),
            ],
        ),
        "products": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _list_like_of_dicts,
                    title="products or a list-like collection of dicts",
                ),
            ],
        ),
    },
    strict=False,
)
