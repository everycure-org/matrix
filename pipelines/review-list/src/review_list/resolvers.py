"""Custom resolvers for the review-list pipeline.

These resolvers provide functionality for environment variable handling,
type casting, and dictionary merging in configuration files.
"""

import os
from copy import deepcopy


def merge_dicts(dict1: dict, dict2: dict) -> dict:
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


def env(key: str, default: str = None, allow_null: str = False) -> str | None:
    """Load a variable from the environment.

    See https://omegaconf.readthedocs.io/en/stable/custom_resolvers.html#custom-resolvers

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
        raise KeyError(
            f"Environment variable '{key}' not found or default value {default} is None"
        )
