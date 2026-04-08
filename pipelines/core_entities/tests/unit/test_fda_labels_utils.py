import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core_entities.utils.fda_labels_utils import (
    DEFAULT_API_KEY_ENV_VAR,
    DrugResult,
    FdaLabelsConfig,
    FdaLabelsQueryConfig,
    build_search_query,
    build_url,
    load_fda_labels_config,
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


def test_build_url_prefers_explicit_api_key_over_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ENV_VAR", "env-key")
    config = FdaLabelsConfig(
        api_url="https://test.example.com/api",
        api_key_env_var="TEST_ENV_VAR",
        api_key="explicit-key",
    )

    url = build_url("aspirin", config=config)

    assert "api_key=explicit-key" in url
    assert "api_key=env-key" not in url


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
