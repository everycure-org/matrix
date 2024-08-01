"""Custom resolvers for Kedro project."""
import os
from typing import Dict, Any, Optional
from copy import deepcopy

from dotenv import load_dotenv

load_dotenv()


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


def env(key: str, default: str = None) -> Optional[str]:
    """Load a variable from the environment.

    See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#custom-resolvers

    Args:
        key (str): Key to load.
        default (str): Default value to use instead

    Returns:
        str: Value of the key
    """
    try:
        print(f"key: {key}, default: {default}")
        value = os.environ.get(key, default)
        if value is None:
            raise KeyError()
        return value
    except KeyError:
        raise KeyError(
            f"Environment variable '{key}' not found or default value {default} is None"
        )
