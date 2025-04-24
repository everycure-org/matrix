import pytest

from matrix.cli_commands.run import _validate_env_vars_for_private_data


@pytest.mark.parametrize(
    "env,env_vars,should_raise_error",
    [
        (
            "prod",
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-prod-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-prod-storage",
                "MLFLOW_URL": "https://mlflow.platform.prod.everycure.org/",
            },
            False,
        ),
        (
            "dev",
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-dev-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
            },
            False,
        ),
        (
            "prod",
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-dev-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
                "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
            },
            True,
        ),
        (
            "dev",
            {
                "RUNTIME_GCP_PROJECT_ID": "mtrx-hub-prod-sms",
                "RUNTIME_GCP_BUCKET": "mtrx-us-central1-hub-prod-storage",
                "MLFLOW_URL": "https://mlflow.platform.prod.everycure.org/",
            },
            True,
        ),
    ],
    ids=["prod_success", "dev_success", "prod_with_dev_vars_error", "dev_with_prod_vars_error"],
)
def test_validate_env_vars(monkeypatch, env, env_vars, should_raise_error):
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    if should_raise_error:
        with pytest.raises(RuntimeError):
            _validate_env_vars_for_private_data(env)
    else:
        _validate_env_vars_for_private_data(env)
