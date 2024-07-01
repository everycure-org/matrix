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


def env(key: str) -> Optional[str]:
    """Load a variable from the environment.

    Args:
        key (str): Key to load.

    Returns:
        dict: Value of the key
    """
    try:
        return os.environ[key]
    except KeyError:
        raise KeyError(f"Environment variable '{key}' not found")
