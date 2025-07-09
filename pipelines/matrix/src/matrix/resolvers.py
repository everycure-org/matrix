import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from matrix.utils.environment import load_environment_variables

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
        # Load globals.yml directly with the registered resolvers
        globals_path = Path("conf/base/globals.yml")
        if globals_path.exists():
            globals_config = OmegaConf.load(globals_path)
        else:
            logging.warning(f"Globals configuration file not found at {globals_path}. Using default values.")
            globals_config = {}

            # Extract bucket configurations from globals
            dev_bucket = globals_config.get("dev_gcs_bucket", "gs://mtrx-us-central1-hub-dev-storage")
            prod_bucket = globals_config.get("prod_gcs_bucket", "gs://mtrx-us-central1-hub-prod-storage")
            public_bucket = globals_config.get("public_gcs_bucket", "gs://data.dev.everycure.org")

        # Importing here to avoid circular import
        # TODO: Refactor to avoid circular import
        from matrix.settings import DYNAMIC_PIPELINES_MAPPING

        pipeline_mapping = DYNAMIC_PIPELINES_MAPPING()
        integration_sources = pipeline_mapping.get("integration", [])

        # Find the source configuration
        path_suffix = "/data/01_RAW"
        include_private = os.getenv("INCLUDE_PRIVATE_DATASETS", "0") == "1"

        for source_config in integration_sources:
            if source_config.get("name") == source_name:
                if source_config.get("is_public", False):
                    bucket = public_bucket
                    bucket_type = "public"
                elif source_config.get("is_private", False) and include_private:
                    bucket = prod_bucket
                    bucket_type = "private"
                else:
                    bucket = dev_bucket
                    bucket_type = "development"

                logging.info(f"Using {bucket_type} bucket for source: {source_name}: {bucket}{path_suffix}")
                return f"{bucket}{path_suffix}"

        # Default to development bucket if source not found
        return f"{dev_bucket}{path_suffix}"

    except Exception as e:
        logging.error(f"Error resolving kg_raw path for source '{source_name}': {e}")
        # If there's any error, default to dev bucket
        dev_bucket = os.getenv("DEV_GCS_BUCKET", "gs://mtrx-hub-dev-3of")
        return f"{dev_bucket}/data/01_RAW"
