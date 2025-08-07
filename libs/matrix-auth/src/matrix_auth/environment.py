from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_environment_variables():
    """Load environment variables from .env.defaults and .env files.

    .env.defaults is loaded first, then .env overwrites any existing values.
    """
    defaults_path = Path(".env.defaults")
    if defaults_path.exists():
        load_dotenv(dotenv_path=defaults_path, override=False)

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
