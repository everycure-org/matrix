import os


def load_env_vars(env):
    """Load environment variables dynamically based on APP_ENV."""

    config = {
        "MLFLOW_URL": os.getenv(f"{env.upper()}_MLFLOW_URL", ""),
    }

    for key, value in config.items():
        os.environ[key] = value
