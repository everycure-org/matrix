"""Custom resolvers for Kedro project."""
from typing import Dict
from copy import deepcopy


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
