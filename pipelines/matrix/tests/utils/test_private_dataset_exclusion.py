import pytest
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
    "runtime_project_id,expected_sources",
    [
        (
            "mtrx-hub-dev-3of",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
        (
            "mtrx-hub-prod-sms",
            [
                {"name": "public_source_1", "integrate_in_kg": True, "is_private": False},
                {"name": "private_source", "integrate_in_kg": True, "is_private": True},
                {"name": "public_source_2", "integrate_in_kg": True, "is_private": False},
            ],
        ),
    ],
    ids=["dev_environment", "prod_environment"],
)
def test_integration_sources_filtering(integration_mapping, monkeypatch, runtime_project_id, expected_sources):
    monkeypatch.setenv("RUNTIME_GCP_PROJECT_ID", runtime_project_id)
    result = disable_private_datasets(integration_mapping)
    result_nested = disable_private_datasets(generate_dynamic_pipeline_mapping(integration_mapping))
    assert result["integration"] == expected_sources
    assert result_nested["integration"] == expected_sources
