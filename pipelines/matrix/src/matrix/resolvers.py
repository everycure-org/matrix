import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv

from matrix.settings import DYNAMIC_PIPELINES_MAPPING


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


# This ensures that environment variables are loaded at module import and thus
# before the pipeline is run or any data is loaded.
load_environment_variables()


def cast_to_int(val: str) -> int:
    """Convert input value into integer.

    This resolver should be used to ensure values extracted from the environment
    are correctly casted to the expected type.

    Args:
       val: value to convert
    Returns:
       Value casted to integer
    """
    return int(val)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge two dictionaries.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary.
    """
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def env(key: str, default: str = None, allow_null: str = False) -> Optional[str]:
    """Load a variable from the environment.

    See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#custom-resolvers

    Args:
        key (str): Key to load.
        default (str): Default value to use instead
        allow_null (bool): Bool indicating whether null is allowed
    Returns:
        str: Value of the key
    """
    try:
        value = os.environ.get(key, default)
        if value is None and not allow_null:
            raise KeyError()
        return value
    except KeyError:
        raise KeyError(f"Environment variable '{key}' not found or default value {default} is None")


def if_null(val: Optional[Any], if_null_val: str, else_val: str):
    """Resolver to conditionally load a configuration entry."""
    return if_null_val if val is None else else_val


def get_bucket_for_source(source_name: str, dev_bucket: str, public_bucket: str) -> str:
    """Get the appropriate bucket path for a data source based on its configuration.

    Args:
        source_name: Name of the data source (e.g., 'rtx_kg2')
        dev_bucket: Development bucket path
        public_bucket: Public bucket path

    Returns:
        str: The bucket path to use for the data source
    """

    try:
        pipeline_mapping = DYNAMIC_PIPELINES_MAPPING()
        integration_sources = pipeline_mapping.get("integration", [])

        # Find the source configuration
        for source_config in integration_sources:
            if source_config.get("name") == source_name:
                # If is_public is True, use public bucket
                if source_config.get("is_public", False):
                    return public_bucket
                # Otherwise, use dev bucket (default behavior)
                return dev_bucket

        # If source not found, default to dev bucket
        return dev_bucket

    except Exception:
        # If there's any error accessing the configuration, default to dev bucket
        return dev_bucket


def get_kg_raw_path_for_source(source_name: str) -> str:
    """Get the appropriate kg_raw path for any data source dynamically.

    This resolver automatically selects between dev, prod, or public buckets
    based on the source configuration (is_private, is_public flags).

    Args:
        source_name: Name of the data source (e.g., 'rtx_kg2', 'robokop')

    Returns:
        str: The complete kg_raw path for the data source
    """

    try:
        # Get bucket configurations from environment/globals
        dev_bucket = os.getenv("DEV_GCS_BUCKET", "gs://mtrx-hub-dev-3of")
        prod_bucket = os.getenv("PROD_GCS_BUCKET", "gs://mtrx-us-central1-hub-prod-storage")
        public_bucket = os.getenv("PUBLIC_GCS_BUCKET", "gs://data.dev.everycure.org")

        pipeline_mapping = DYNAMIC_PIPELINES_MAPPING()
        integration_sources = pipeline_mapping.get("integration", [])

        # Find the source configuration
        for source_config in integration_sources:
            if source_config.get("name") == source_name:
                # Priority: is_public > is_private > default (dev)
                if source_config.get("is_public", False):
                    return f"{public_bucket}/data/01_RAW"
                elif source_config.get("is_private", False):
                    return f"{prod_bucket}/data/01_RAW"
                else:
                    return f"{dev_bucket}/data/01_RAW"

        # If source not found, default to dev bucket
        return f"{dev_bucket}/data/01_RAW"

    except Exception:
        # If there's any error, default to dev bucket
        dev_bucket = os.getenv("DEV_GCS_BUCKET", "gs://mtrx-hub-dev-3of")
        return f"{dev_bucket}/data/01_RAW"
