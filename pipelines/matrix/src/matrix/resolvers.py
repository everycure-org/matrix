import os
from copy import deepcopy
from typing import Any, Dict, Optional

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
