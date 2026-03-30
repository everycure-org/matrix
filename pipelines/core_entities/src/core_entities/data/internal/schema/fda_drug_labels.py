"""Pandera schema definitions for drugs@FDA label matching pipeline nodes."""

from __future__ import annotations

from numbers import Integral

import pandera.pandas as pa


def _is_list(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell must be a list."""

    def _is_list_like(value: object) -> bool:
        if isinstance(value, list):
            return True
        if isinstance(value, (tuple, set)):
            return True
        if hasattr(value, "tolist") and not isinstance(value, (dict, str, bytes)):
            return isinstance(value.tolist(), list)
        return False

    return series.apply(_is_list_like)


def _is_non_negative_integer(series: pa.typing.Series) -> pa.typing.Series:
    """Each cell must be a non-negative integer."""
    return series.apply(lambda x: isinstance(x, Integral) and x >= 0)


CURATED_DRUG_LIST_FOR_FDA_MATCH_SCHEMA = pa.DataFrameSchema(
    columns={
        "id": pa.Column(nullable=False),
        "name": pa.Column(nullable=False),
        "available_in_combo_with": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="available_in_combo_with must be a list",
                )
            ],
        ),
    },
    strict=False,
)


FDA_DRUG_LABELS_UNFILTERED_SCHEMA = pa.DataFrameSchema(
    columns={
        "fda_rows": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="fda_rows must be a list")],
        ),
        "fda_match_count": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="fda_match_count must be a non-negative integer",
                )
            ],
        ),
        "drug_name": pa.Column(nullable=False),
        "id": pa.Column(nullable=False),
        "search_terms": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="search_terms must be a list")],
        ),
        "available_in_combo_with": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="available_in_combo_with must be a list",
                )
            ],
        ),
    },
    strict=True,
)


FDA_DRUG_LABELS_FILTERED_PARQUET_SCHEMA = pa.DataFrameSchema(
    columns={
        "fda_rows": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="fda_rows must be a list")],
        ),
        "fda_match_count": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="fda_match_count must be a non-negative integer",
                )
            ],
        ),
        "drug_name": pa.Column(nullable=False),
        "id": pa.Column(nullable=False),
        "search_terms": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="search_terms must be a list")],
        ),
        "available_in_combo_with": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="available_in_combo_with must be a list",
                )
            ],
        ),
        "filtered_fda_values": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="filtered_fda_values must be a list",
                )
            ],
        ),
        "filtered_fda_values_count": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="filtered_fda_values_count must be a non-negative integer",
                )
            ],
        ),
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "is_biologics": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "brand_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="brand_name must be a list")],
        ),
        "generic_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="generic_name must be a list")],
        ),
        "substance_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="substance_name must be a list")],
        ),
        "active_ingredients": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="active_ingredients must be a list",
                )
            ],
        ),
        "marketing_status": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="marketing_status must be a list",
                )
            ],
        ),
        "is_anda": pa.Column(
            nullable=False,
            dtype=bool,
        ),
    },
    strict=True,
)


FDA_DRUG_LABELS_FILTERED_TSV_SCHEMA = pa.DataFrameSchema(
    columns={
        "fda_match_count": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="fda_match_count must be a non-negative integer",
                )
            ],
        ),
        "drug_name": pa.Column(nullable=False),
        "id": pa.Column(nullable=False),
        "search_terms": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="search_terms must be a list")],
        ),
        "available_in_combo_with": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="available_in_combo_with must be a list",
                )
            ],
        ),
        "filtered_fda_values_count": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="filtered_fda_values_count must be a non-negative integer",
                )
            ],
        ),
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "is_biologics": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "brand_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="brand_name must be a list")],
        ),
        "generic_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="generic_name must be a list")],
        ),
        "substance_name": pa.Column(
            nullable=True,
            checks=[pa.Check(_is_list, title="substance_name must be a list")],
        ),
        "active_ingredients": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="active_ingredients must be a list",
                )
            ],
        ),
        "marketing_status": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="marketing_status must be a list",
                )
            ],
        ),
        "is_anda": pa.Column(
            nullable=False,
            dtype=bool,
        ),
    },
    strict=True,
)


FDA_DRUG_LABELS_FOR_BIOSIMILAR_INPUT_SCHEMA = pa.DataFrameSchema(
    columns={
        "is_biologics": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "filtered_fda_values": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="filtered_fda_values must be a list when present",
                )
            ],
        ),
        "fda_rows": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="fda_rows must be a list when present",
                )
            ],
        ),
    },
    strict=False,
    checks=[
        pa.Check(
            lambda df: ("filtered_fda_values" in df.columns) or ("fda_rows" in df.columns),
            title="one of filtered_fda_values or fda_rows must be present",
        )
    ],
)


FDA_DRUG_LABELS_BIOSIMILAR_PARQUET_SCHEMA = pa.DataFrameSchema(
    columns={
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "biosimilar_bla_types": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="biosimilar_bla_types must be a list",
                )
            ],
        ),
        "biosimilar_application_numbers": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="biosimilar_application_numbers must be a list",
                )
            ],
        ),
        "fda_rows": pa.Column(
            required=False,
            nullable=True,
            checks=[pa.Check(_is_list, title="fda_rows must be a list when present")],
        ),
        "filtered_fda_values": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="filtered_fda_values must be a list when present",
                )
            ],
        ),
    },
    strict=False,
)


FDA_DRUG_LABELS_BIOSIMILAR_TSV_SCHEMA = pa.DataFrameSchema(
    columns={
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "biosimilar_bla_types": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="biosimilar_bla_types must be a list",
                )
            ],
        ),
        "biosimilar_application_numbers": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="biosimilar_application_numbers must be a list",
                )
            ],
        ),
    },
    strict=False,
)


FDA_DRUG_LABELS_FOR_OTC_INPUT_SCHEMA = pa.DataFrameSchema(
    columns={
        "drug_name": pa.Column(nullable=False),
        "is_fda_generic_drug": pa.Column(
            required=False,
            nullable=False,
            dtype=bool,
        ),
        "marketing_status": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="marketing_status must be a list when present",
                )
            ],
        ),
        "fda_rows": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="fda_rows must be a list when present",
                )
            ],
        ),
        "filtered_fda_values": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="filtered_fda_values must be a list when present",
                )
            ],
        ),
    },
    strict=False,
)


FDA_DRUG_LABELS_OTC_PARQUET_SCHEMA = pa.DataFrameSchema(
    columns={
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "otc_monograph_checked": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "otc_monograph_status": pa.Column(nullable=False),
        "otc_monograph_application_numbers": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="otc_monograph_application_numbers must be a list",
                )
            ],
        ),
        "otc_monograph_total_matches": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="otc_monograph_total_matches must be a non-negative integer",
                )
            ],
        ),
        "otc_monograph_error_msg": pa.Column(nullable=False),
        "is_otc_monograph": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "marketing_status": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="marketing_status must be a list when present",
                )
            ],
        ),
        "fda_rows": pa.Column(
            required=False,
            nullable=True,
            checks=[pa.Check(_is_list, title="fda_rows must be a list when present")],
        ),
        "filtered_fda_values": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="filtered_fda_values must be a list when present",
                )
            ],
        ),
    },
    strict=False,
)


FDA_DRUG_LABELS_OTC_TSV_SCHEMA = pa.DataFrameSchema(
    columns={
        "is_fda_generic_drug": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "otc_monograph_checked": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "otc_monograph_status": pa.Column(nullable=False),
        "otc_monograph_application_numbers": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="otc_monograph_application_numbers must be a list",
                )
            ],
        ),
        "otc_monograph_total_matches": pa.Column(
            nullable=True,
            checks=[
                pa.Check(
                    _is_non_negative_integer,
                    title="otc_monograph_total_matches must be a non-negative integer",
                )
            ],
        ),
        "otc_monograph_error_msg": pa.Column(nullable=False),
        "is_otc_monograph": pa.Column(
            nullable=False,
            dtype=bool,
        ),
        "marketing_status": pa.Column(
            required=False,
            nullable=True,
            checks=[
                pa.Check(
                    _is_list,
                    title="marketing_status must be a list when present",
                )
            ],
        ),
    },
    strict=False,
)
