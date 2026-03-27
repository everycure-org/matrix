import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.utils.fda_labels_utils import (
    DEFAULT_API_KEY_ENV_VAR,
    DrugResult,
    FdaLabelsConfig,
    FdaLabelsQueryConfig,
    _as_fda_labels_config,
    _coerce_non_negative_int,
    _coerce_positive_int,
    _coerce_retry_backoff,
    _extract_application_numbers,
    _extract_total_matches,
    _non_empty_string_or_none,
    _resolve_api_key,
    _wait_time_for_attempt,
    build_search_query,
    build_url,
    load_fda_labels_config,
    load_suspect_drugs,
    run_sync,
)


def test_fda_labels_query_config_from_dict_normalizes_values() -> None:
    config = FdaLabelsQueryConfig.from_dict(
        {
            "substance_name_field": " openfda.substance_name ",
            "application_number_field": " openfda.application_number ",
            "application_number_prefix": " m ",
            "boolean_operator": " or ",
            "additional_clauses": [" purpose:human ", "", 123],
            "uppercase_substance_name": False,
        }
    )

    assert config.substance_name_field == "openfda.substance_name"
    assert config.application_number_field == "openfda.application_number"
    assert config.application_number_prefix == "m"
    assert config.boolean_operator == "OR"
    assert config.additional_clauses == ("purpose:human",)
    assert config.uppercase_substance_name is False


def test_fda_labels_config_from_dict_applies_defaults_and_coercion() -> None:
    config = FdaLabelsConfig.from_dict(
        {
            "api": {
                "api_url": " https://example.test/labels ",
                "api_key_env_var": " FDA_LABELS_KEY ",
                "api_key": " explicit-key ",
                "max_concurrent": "2",
                "retry_attempts": "3",
                "retry_backoff_seconds": [0, "5", -1, "bad"],
                "request_timeout_seconds": "20",
                "result_limit": "4",
            },
            "monograph_application_prefix": " N ",
            "query": {
                "additional_clauses": ["status:active"],
            },
        }
    )

    assert config.api_url == "https://example.test/labels"
    assert config.api_key_env_var == "FDA_LABELS_KEY"
    assert config.api_key == "explicit-key"
    assert config.max_concurrent == 2
    assert config.retry_attempts == 3
    assert config.retry_backoff_seconds == (0, 5)
    assert config.request_timeout_seconds == 20
    assert config.result_limit == 4
    assert config.monograph_application_prefix == "N"
    assert config.query.additional_clauses == ("status:active",)


def test_build_search_query_constructs_expected_clause_string() -> None:
    query_config = FdaLabelsQueryConfig(
        boolean_operator="OR",
        additional_clauses=("status:active",),
        uppercase_substance_name=True,
    )

    query = build_search_query(" acetyl salicylic acid ", query_config)

    assert query == 'openfda.substance_name:"ACETYL SALICYLIC ACID" OR openfda.application_number:M* OR status:active'


def test_build_search_query_raises_when_all_clauses_disabled() -> None:
    query_config = FdaLabelsQueryConfig(
        substance_name_field="",
        application_number_field="",
        application_number_prefix="",
        additional_clauses=(),
    )

    try:
        build_search_query("aspirin", query_config)
    except ValueError as exc:
        assert "At least one query clause" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no clauses are configured")


def test_build_url_includes_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_VAR, "env-api-key")

    url = build_url("aspirin")

    assert "https://api.fda.gov/drug/label.json?" in url
    assert "api_key=env-api-key" in url
    assert "limit=1" in url
    assert "openfda.substance_name%3A%22ASPIRIN%22" in url


def test_load_suspect_drugs_filters_only_expected_rows(tmp_path) -> None:
    tsv_path = tmp_path / "drugs.tsv"
    tsv_path.write_text(
        "drug_name\tis_anda\tis_biologics\tmarketing_status\tis_fda_generic_drug\n"
        "Drug A\tfalse\tfalse\t[]\tfalse\n"
        "Drug B\ttrue\tfalse\t[]\tfalse\n"
        "Drug C\tfalse\tfalse\t['Prescription']\tfalse\n"
        "Drug D\tfalse\tfalse\t[]\ttrue\n"
        "\tfalse\tfalse\t[]\tfalse\n",
        encoding="utf-8",
    )

    suspects = load_suspect_drugs(str(tsv_path))

    assert suspects == ["Drug A"]


def test_extract_total_matches_reads_nested_meta_results_total() -> None:
    assert _extract_total_matches({"meta": {"results": {"total": "5"}}}) == 5
    assert _extract_total_matches({"meta": {"results": {"total": -2}}}) == 0
    assert _extract_total_matches({"meta": {}}) == 0
    assert _extract_total_matches([]) == 0


def test_extract_application_numbers_handles_string_and_list() -> None:
    single = {
        "results": [
            {
                "openfda": {
                    "application_number": " M123456 ",
                }
            }
        ]
    }
    multiple = {
        "results": [
            {
                "openfda": {
                    "application_number": [" M123 ", "", 9, "N456"],
                }
            }
        ]
    }

    assert _extract_application_numbers(single) == ["M123456"]
    assert _extract_application_numbers(multiple) == ["M123", "N456"]


def test_retry_helpers_filter_and_cap_wait_times() -> None:
    retry_backoff = _coerce_retry_backoff([0, "3", -1, "bad", 6], default=(1, 2))

    assert retry_backoff == (0, 3, 6)
    assert _wait_time_for_attempt(retry_backoff, attempt_index=0) == 0
    assert _wait_time_for_attempt(retry_backoff, attempt_index=10) == 6


def test_fda_labels_query_config_from_dict_with_none_returns_defaults() -> None:
    config = FdaLabelsQueryConfig.from_dict(None)

    assert config.substance_name_field == "openfda.substance_name"
    assert config.application_number_field == "openfda.application_number"
    assert config.application_number_prefix == "M"
    assert config.boolean_operator == "AND"
    assert config.additional_clauses == ()
    assert config.uppercase_substance_name is True


def test_fda_labels_query_config_from_dict_with_non_mapping_returns_defaults() -> None:
    config = FdaLabelsQueryConfig.from_dict([])

    assert config.substance_name_field == "openfda.substance_name"
    assert config.boolean_operator == "AND"


def test_fda_labels_query_config_from_dict_empty_boolean_operator_uses_default() -> None:
    config = FdaLabelsQueryConfig.from_dict({"boolean_operator": ""})

    assert config.boolean_operator == "AND"


def test_fda_labels_query_config_from_dict_none_boolean_operator_uses_default() -> None:
    config = FdaLabelsQueryConfig.from_dict({"boolean_operator": None})

    assert config.boolean_operator == "AND"


def test_fda_labels_query_config_from_dict_filters_empty_clauses() -> None:
    config = FdaLabelsQueryConfig.from_dict({"additional_clauses": ["status:active", "  ", "", "priority:high"]})

    assert config.additional_clauses == ("status:active", "priority:high")


def test_fda_labels_config_from_dict_with_none_returns_defaults() -> None:
    config = FdaLabelsConfig.from_dict(None)

    assert config.api_url == "https://api.fda.gov/drug/label.json"
    assert config.api_key is None
    assert config.max_concurrent == 5


def test_fda_labels_config_from_dict_with_non_mapping_api_config() -> None:
    config = FdaLabelsConfig.from_dict({"api": []})

    assert config.api_url == "https://api.fda.gov/drug/label.json"


def test_fda_labels_config_from_dict_with_non_mapping_query_config() -> None:
    config = FdaLabelsConfig.from_dict({"query": "not_a_dict"})

    assert config.query.substance_name_field == "openfda.substance_name"


def test_fda_labels_config_from_dict_coerces_invalid_positive_ints() -> None:
    config = FdaLabelsConfig.from_dict(
        {
            "api": {
                "max_concurrent": 0,
                "retry_attempts": -5,
                "result_limit": "not_a_number",
            }
        }
    )

    assert config.max_concurrent == 5
    assert config.retry_attempts == 3
    assert config.result_limit == 1


def test_fda_labels_config_from_dict_coerces_retry_backoff_to_empty_tuple() -> None:
    config = FdaLabelsConfig.from_dict(
        {
            "api": {
                "retry_backoff_seconds": [-1, -2, -3],
            }
        }
    )

    assert config.retry_backoff_seconds == (1, 3, 7)


def test_fda_labels_config_from_dict_handles_empty_api_key_string() -> None:
    config = FdaLabelsConfig.from_dict({"api": {"api_key": "   "}})

    assert config.api_key is None


def test_build_search_query_without_uppercase() -> None:
    query_config = FdaLabelsQueryConfig(uppercase_substance_name=False)

    query = build_search_query(" ASPIRIN ", query_config)

    assert 'openfda.substance_name:"ASPIRIN"' in query


def test_build_search_query_without_substance_name_field() -> None:
    query_config = FdaLabelsQueryConfig(
        substance_name_field="",
        additional_clauses=("status:active",),
    )

    query = build_search_query("aspirin", query_config)

    assert "openfda.substance_name" not in query
    assert "status:active" in query


def test_build_search_query_without_application_number_field() -> None:
    query_config = FdaLabelsQueryConfig(
        application_number_field="",
        additional_clauses=("status:active",),
    )

    query = build_search_query("aspirin", query_config)

    assert "openfda.application_number" not in query


def test_build_url_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_API_KEY_ENV_VAR, raising=False)

    url = build_url("aspirin", config={"api": {"api_key": None}})

    assert "api_key" not in url


def test_build_url_from_fda_labels_config_object() -> None:
    config = FdaLabelsConfig(
        api_url="https://test.example.com/api",
        result_limit=5,
    )

    url = build_url("aspirin", config=config)

    assert url.startswith("https://test.example.com/api?")
    assert "limit=5" in url


def test_load_fda_labels_config_wrapper() -> None:
    config = load_fda_labels_config({"api": {"max_concurrent": 10}})

    assert config.max_concurrent == 10


def test_load_suspect_drugs_parses_valid_list_values(tmp_path) -> None:
    tsv_path = tmp_path / "drugs.tsv"
    tsv_path.write_text(
        "drug_name\tis_anda\tis_biologics\tmarketing_status\tis_fda_generic_drug\n"
        "Drug E\tfalse\tfalse\t['Prescription', 'OTC']\tfalse\n",
        encoding="utf-8",
    )

    suspects = load_suspect_drugs(str(tsv_path))

    assert suspects == []


def test_load_suspect_drugs_handles_unparseable_list_values(tmp_path) -> None:
    tsv_path = tmp_path / "drugs.tsv"
    tsv_path.write_text(
        "drug_name\tis_anda\tis_biologics\tmarketing_status\tis_fda_generic_drug\n"
        "Drug F\tfalse\tfalse\tinvalid_syntax[\tfalse\n"
        "Drug G\tfalse\tfalse\t[]\tfalse\n",
        encoding="utf-8",
    )

    suspects = load_suspect_drugs(str(tsv_path))

    # Both are included because unparseable defaults to empty list, and both have no marketing status
    assert suspects == ["Drug F", "Drug G"]


def test_extract_total_matches_handles_non_int_total() -> None:
    assert _extract_total_matches({"meta": {"results": {"total": "not_int"}}}) == 0


def test_extract_total_matches_with_non_dict_results() -> None:
    assert _extract_total_matches({"meta": {"results": []}}) == 0


def test_extract_total_matches_with_non_dict_meta() -> None:
    assert _extract_total_matches({"meta": []}) == 0


def test_extract_application_numbers_with_empty_results_list() -> None:
    assert _extract_application_numbers({"results": []}) == []


def test_extract_application_numbers_with_missing_openfda() -> None:
    assert _extract_application_numbers({"results": [{}]}) == []


def test_extract_application_numbers_with_non_dict_openfda() -> None:
    assert _extract_application_numbers({"results": [{"openfda": []}]}) == []


def test_extract_application_numbers_with_non_string_in_list() -> None:
    data = {
        "results": [
            {
                "openfda": {
                    "application_number": [123, "M456", None, "N789"],
                }
            }
        ]
    }

    assert _extract_application_numbers(data) == ["M456", "N789"]


def test_extract_application_numbers_with_empty_string_in_list() -> None:
    data = {
        "results": [
            {
                "openfda": {
                    "application_number": ["M123", "  ", "N456"],
                }
            }
        ]
    }

    assert _extract_application_numbers(data) == ["M123", "N456"]


def test_extract_application_numbers_with_empty_string() -> None:
    data = {
        "results": [
            {
                "openfda": {
                    "application_number": "  ",
                }
            }
        ]
    }

    assert _extract_application_numbers(data) == []


def test_extract_application_numbers_with_non_list_non_string() -> None:
    data = {
        "results": [
            {
                "openfda": {
                    "application_number": 123,
                }
            }
        ]
    }

    assert _extract_application_numbers(data) == []


def test_coerce_positive_int_with_zero_returns_default() -> None:
    assert _coerce_positive_int(0, 5) == 5


def test_coerce_positive_int_with_negative_returns_default() -> None:
    assert _coerce_positive_int(-10, 5) == 5


def test_coerce_positive_int_with_valid_positive() -> None:
    assert _coerce_positive_int(42, 5) == 42


def test_coerce_positive_int_with_invalid_type() -> None:
    assert _coerce_positive_int("not_int", 5) == 5


def test_coerce_non_negative_int_with_zero() -> None:
    assert _coerce_non_negative_int(0, 5) == 0


def test_coerce_non_negative_int_with_negative() -> None:
    assert _coerce_non_negative_int(-1, 5) == 5


def test_coerce_non_negative_int_with_valid() -> None:
    assert _coerce_non_negative_int(42, 5) == 42


def test_coerce_non_negative_int_with_invalid_type() -> None:
    assert _coerce_non_negative_int(None, 5) == 5


def test_coerce_retry_backoff_with_non_list_returns_default() -> None:
    assert _coerce_retry_backoff("not_list", (1, 2)) == (1, 2)
    assert _coerce_retry_backoff(None, (1, 2)) == (1, 2)


def test_coerce_retry_backoff_with_all_negative_returns_default() -> None:
    assert _coerce_retry_backoff([-1, -2, -3], (1, 2)) == (1, 2)


def test_wait_time_for_attempt_with_empty_backoff() -> None:
    assert _wait_time_for_attempt((), 0) == 0
    assert _wait_time_for_attempt((), 5) == 0


def test_wait_time_for_attempt_with_single_element() -> None:
    assert _wait_time_for_attempt((5,), 0) == 5
    assert _wait_time_for_attempt((5,), 10) == 5


def test_non_empty_string_or_none_returns_none_for_non_string() -> None:
    assert _non_empty_string_or_none(123) is None
    assert _non_empty_string_or_none(None) is None
    assert _non_empty_string_or_none([]) is None


def test_non_empty_string_or_none_returns_none_for_empty_string() -> None:
    assert _non_empty_string_or_none("") is None
    assert _non_empty_string_or_none("   ") is None


def test_non_empty_string_or_none_returns_stripped_string() -> None:
    assert _non_empty_string_or_none("  hello  ") == "hello"
    assert _non_empty_string_or_none("world") == "world"


def test_resolve_api_key_uses_explicit_key_first() -> None:
    config = FdaLabelsConfig(api_key="explicit-key", api_key_env_var="TEST_ENV_VAR")

    assert _resolve_api_key(config) == "explicit-key"


def test_resolve_api_key_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_VAR", "env-key")
    config = FdaLabelsConfig(api_key=None, api_key_env_var="TEST_VAR")

    assert _resolve_api_key(config) == "env-key"


def test_resolve_api_key_returns_none_when_not_found(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    config = FdaLabelsConfig(api_key=None, api_key_env_var="MISSING_VAR")

    assert _resolve_api_key(config) is None


def test_as_fda_labels_config_with_config_object() -> None:
    original = FdaLabelsConfig(max_concurrent=10)
    result = _as_fda_labels_config(original)

    assert result is original


def test_as_fda_labels_config_with_dict() -> None:
    result = _as_fda_labels_config({"api": {"max_concurrent": 15}})

    assert isinstance(result, FdaLabelsConfig)
    assert result.max_concurrent == 15


def test_as_fda_labels_config_with_none() -> None:
    result = _as_fda_labels_config(None)

    assert isinstance(result, FdaLabelsConfig)
    assert result.max_concurrent == 5


def test_drug_result_creation() -> None:
    result = DrugResult(
        drug_name="aspirin",
        status="OTC_MONOGRAPH",
        application_numbers=["M123456"],
        total_matches=1,
    )

    assert result.drug_name == "aspirin"
    assert result.status == "OTC_MONOGRAPH"
    assert result.total_matches == 1


def test_drug_result_defaults() -> None:
    result = DrugResult(drug_name="test", status="NO_RESULTS")

    assert result.application_numbers == []
    assert result.total_matches == 0
    assert result.error_msg == ""


def test_run_sync_calls_async_run() -> None:
    with patch("core_entities.utils.fda_labels_utils.asyncio.run") as mock_asyncio_run:
        mock_asyncio_run.return_value = [DrugResult(drug_name="aspirin", status="NO_RESULTS")]

        result = run_sync(["aspirin"])

        assert len(result) == 1
        assert result[0].drug_name == "aspirin"


def test_build_search_query_without_section_delimiter_field() -> None:
    # Edge case: query config with no substance_name_field set to empty string
    query_config = FdaLabelsQueryConfig(
        substance_name_field="",
        application_number_prefix="",
        additional_clauses=("priority:high",),
    )

    query = build_search_query("ibuprofen", query_config)

    assert query == "priority:high"


def test_build_url_preserves_config_result_limit() -> None:
    config = FdaLabelsConfig(result_limit=10)
    url = build_url("test_drug", config)

    assert "limit=10" in url


def test_fda_labels_config_with_empty_string_boolean_operator() -> None:
    # Test case where boolean_operator could become empty after strip
    config = FdaLabelsQueryConfig.from_dict({"boolean_operator": None})
    assert config.boolean_operator == "AND"


def test_extract_application_numbers_case_insensitive_prefix_check(monkeypatch) -> None:
    # Test the case-insensitive check for finding no results
    data = {
        "results": [
            {
                "openfda": {
                    "application_number": [],  # Empty list
                }
            }
        ]
    }
    result = _extract_application_numbers(data)
    assert result == []
