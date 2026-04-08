import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.pipelines.drug_list.nodes import (
    ingest_fda_drug_json,
    ingest_fda_purple_book_data,
    resolve_fda_drugs_matches_to_drug_list_filtered,
    resolve_fda_drugs_that_are_biosimilar_and_are_generic,
)
from core_entities.utils.fda_drugs_utils import (
    extract_openfda_field,
    extract_product_active_ingredients,
    extract_product_marketing_status,
    filter_fda_rows,
    has_anda_application_number,
    match_drug_to_fda_worker,
    normalize_fda_results_to_dataframe,
)


def test_match_drug_to_fda_worker_uses_configured_openfda_paths() -> None:
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

    work_item = {
        "name": "brand lookup",
        "id": "EC:00001",
        "search_terms": {"brand a"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["openfda.brand_name"],
    )

    assert result["fda_match_count"] == 1
    assert result["fda_rows"][0]["application_number"] == "ANDA111111"


def test_match_drug_to_fda_worker_supports_products_active_ingredients_path() -> None:
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

    work_item = {
        "name": "active ingredient lookup",
        "id": "EC:00002",
        "search_terms": {"helper y"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=["products.active_ingredients.name"],
    )

    assert result["fda_match_count"] == 1
    assert result["fda_rows"][0]["application_number"] == "BLA123456"


def test_match_drug_to_fda_worker_ignores_non_string_and_blank_column_paths() -> None:
    fda_drug_list = pd.DataFrame(
        [
            {
                "application_number": "ANDA101010",
                "openfda": {"brand_name": ["Brand A"]},
                "products": [],
            }
        ]
    )

    work_item = {
        "name": "brand lookup",
        "id": "EC:00003",
        "search_terms": {"brand a"},
        "available_in_combo_with": [],
    }

    result = match_drug_to_fda_worker(
        work_item=work_item,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=[123, "   ", "openfda.brand_name"],  # type: ignore[list-item]
    )

    assert result["fda_match_count"] == 1
    assert result["fda_rows"][0]["application_number"] == "ANDA101010"


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
            "bla_number": ["761071"],
            "bla_type": ["351(k) Interchangeable"],
        }
    )
    params = {"generic_status_bla_type": ["351(k) Interchangeable"]}

    result_parquet, _ = resolve_fda_drugs_that_are_biosimilar_and_are_generic(df, params, purple_book_df)

    assert result_parquet["biosimilar_application_numbers"].tolist() == [["bla761071"]]
    assert result_parquet["biosimilar_bla_types"].tolist() == [["351(k) Interchangeable"]]
    assert result_parquet["is_fda_generic_drug"].tolist() == [True]


def test_ingest_fda_purple_book_data_normalizes_and_deduplicates_rows() -> None:
    raw_purple_book_df = pd.DataFrame(
        {
            "BLA Number": [761071, "BLA 761071", "761071", ""],
            "BLA Type": ["351(k) Interchangeable", "351(k) Interchangeable", "351(k) Biosimilar", "ignored"],
            "Proper Name": ["adalimumab", "adalimumab", "adalimumab", "invalid"],
        }
    )

    ingested = ingest_fda_purple_book_data(raw_purple_book_df)

    assert list(ingested.columns) == ["bla_number", "bla_type"]
    assert ingested.to_dict(orient="records") == [
        {"bla_number": "761071", "bla_type": "351(k) Interchangeable"},
        {"bla_number": "761071", "bla_type": "351(k) Biosimilar"},
    ]


def test_normalize_fda_results_to_dataframe_normalizes_openfda_known_list_fields() -> None:
    fda_json = {
        "results": [
            {
                "application_number": "ANDA123456",
                "sponsor_name": "Acme Pharma",
                "submissions": [],
                "openfda": {
                    "brand_name": " Brand X ",
                    "generic_name": np.array(["Drug X", "Drug X", ""], dtype=object),
                    "route": "ORAL",
                },
                "products": [],
            }
        ]
    }

    normalized_df = normalize_fda_results_to_dataframe(fda_json)
    normalized_openfda = normalized_df.loc[0, "openfda"]

    assert normalized_df.loc[0, "application_number"] == "anda123456"
    assert normalized_df.loc[0, "sponsor_name"] == "acme pharma"
    assert normalized_openfda["brand_name"] == ["brand x"]
    assert normalized_openfda["generic_name"] == ["drug x"]
    assert normalized_openfda["route"] == ["oral"]


def test_ingest_fda_drug_json_validates_input_and_returns_normalized_dataframe() -> None:
    fda_json_df = pd.DataFrame(
        [
            {
                "application_number": "ANDA123456",
                "sponsor_name": "Acme Pharma",
                "submissions": [],
                "openfda": {"brand_name": " Brand X "},
                "products": [],
            }
        ]
    )

    result = ingest_fda_drug_json(fda_json_df)

    assert result.loc[0, "application_number"] == "anda123456"
    assert result.loc[0, "sponsor_name"] == "acme pharma"
    assert result.loc[0, "openfda"]["brand_name"] == ["brand x"]


def test_ingest_fda_drug_json_raises_schema_error_for_invalid_input() -> None:
    invalid_fda_json_df = pd.DataFrame([{"sponsor_name": "Acme Pharma", "openfda": {}, "products": []}])

    with pytest.raises(pa.errors.SchemaError):
        ingest_fda_drug_json(invalid_fda_json_df)


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


def test_extract_openfda_field_handles_numpy_array_values() -> None:
    rows = [{"openfda": {"brand_name": np.array(["Brand X", "Brand X"], dtype=object)}}]
    result = extract_openfda_field(rows, "brand_name")
    assert result == ["Brand X"]


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
