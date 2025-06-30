from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from matrix.cli_commands.run import _validate_env_vars_for_private_data


@pytest.mark.parametrize(
    "env_vars,should_prompt",
    [
        (
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-prod-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-prod-storage",
                "MLFLOW_URL": "https://mlflow.platform.prod.everycure.org/",
                "INCLUDE_PRIVATE_DATASETS": "1",
            },
            False,
        ),
        (
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-dev-3of",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
            },
            False,
        ),
        (
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-dev-3of",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
                "INCLUDE_PRIVATE_DATASETS": "0",
            },
            False,
        ),
        (
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-dev-3of",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
                "INCLUDE_PRIVATE_DATASETS": "1",
            },
            True,
        ),
        (
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-prod-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.prod.everycure.org/",
                "INCLUDE_PRIVATE_DATASETS": "1",
            },
            True,
        ),
    ],
    ids=[
        "prod_success",
        "dev_success",
        "dev_with_0_flag_success",
        "dev_with_1_flag_error",
        "prod_with_dev_bucket_error",
    ],
)
def test_validate_env_vars(monkeypatch, env_vars, should_prompt):
    # Unless you do this, and you have that value in your .env, it is still picked up by the test.
    # This is why you must explicitly delete it, before potentially setting it via monkeypatch, if you want to
    # test the scenario where that value is absent as part of the test requirements:
    monkeypatch.delenv("INCLUDE_PRIVATE_DATASETS", raising=False)

    # Set INCLUDE_PRIVATE_DATASETS if it's in the test parameters
    if "INCLUDE_PRIVATE_DATASETS" in env_vars:
        monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", env_vars["INCLUDE_PRIVATE_DATASETS"])

    # Mock the runtime functions to return the test values
    with patch("matrix.cli_commands.run.get_runtime_gcp_project_id") as mock_project, patch(
        "matrix.cli_commands.run.get_runtime_gcp_bucket"
    ) as mock_bucket, patch("matrix.cli_commands.run.get_runtime_mlflow_url") as mock_mlflow:
        mock_project.return_value = env_vars["RUNTIME_GCP_PROJECT_ID"]
        mock_bucket.return_value = env_vars["RUNTIME_GCP_BUCKET"]
        mock_mlflow.return_value = env_vars["MLFLOW_URL"]

        runner = CliRunner()

        if should_prompt:
            with runner.isolation(input="n\n"):
                with pytest.raises(click.exceptions.Abort):
                    _validate_env_vars_for_private_data()
        else:
            with runner.isolation():
                _validate_env_vars_for_private_data()  # Should complete without prompting
