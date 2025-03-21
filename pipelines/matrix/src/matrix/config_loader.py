import os


def load_env_vars(env):
    """Load environment variables dynamically based on APP_ENV."""

    config = {
        "prod": {
            "MLFLOW_URL": "https://mlflow.platform.prod.everycure.org/",
            "GCP_PROJECT_ID": "mtrx-hub-prod-sms",
            "GCP_BUCKET": "mtrx-us-central1-hub-prod-storage",
        },
        "dev": {
            "MLFLOW_URL": "https://mlflow.platform.dev.everycure.org/",
            "GCP_PROJECT_ID": "mtrx-hub-dev-3of",
            "GCP_BUCKET": "mtrx-us-central1-hub-dev-storage",
        },
    }

    current_env = os.environ["GCP_ENV"]

    for key, value in config[current_env].items():
        os.environ[key] = value
