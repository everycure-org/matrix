import pytest

from matrix.argo import get_include_private_datasets_flag
from matrix.utils.hook_utilities import disable_private_datasets, generate_dynamic_pipeline_mapping


@pytest.fixture
def integration_mapping():
    return {
        "integration": [
            {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
            {"name": "private_source", "integrate_in_kg": True, "is_private": True},
            {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
        ]
    }


@pytest.mark.parametrize(
    "include_private_datasets,expected_sources",
    [
        (
            "",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
        (
            "1",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "private_source", "integrate_in_kg": True, "is_private": True},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
    ],
    ids=["dev_environment", "prod_environment"],
)
def test_integration_sources_filtering(integration_mapping, monkeypatch, include_private_datasets, expected_sources):
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", include_private_datasets)
    result = disable_private_datasets(integration_mapping)
    result_nested = disable_private_datasets(generate_dynamic_pipeline_mapping(integration_mapping))
    assert result["integration"] == expected_sources
    assert result_nested["integration"] == expected_sources


@pytest.mark.parametrize(
    "include_private_datasets,expected_flag",
    [
        ("", "false"),
        ("0", "false"),
        ("1", "true"),
    ],
    ids=["env_not_set", "dev_environment", "prod_environment"],
)
def test_get_include_private_datasets_flag(monkeypatch, include_private_datasets, expected_flag):
    if include_private_datasets == "":
        monkeypatch.delenv("INCLUDE_PRIVATE_DATASETS", raising=False)
    else:
        monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", include_private_datasets)

    result = get_include_private_datasets_flag()
    assert result == expected_flag
