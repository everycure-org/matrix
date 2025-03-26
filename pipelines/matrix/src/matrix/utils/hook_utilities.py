import ast
import os
from typing import Any, Dict, List, Optional, Union


def determine_hooks_to_execute(hooks: Dict[str, Any]):
    """Utility that we added to disable hooks through environment variables.

    It looks for env variables that start with KEDRO_HOOKS_DISABLE_ and if one is found, it will disable the hook.

    Parameters:
        hooks: A dictionary of hooks, as defined in settings.py, some of which we want to disable.

    Returns:
        A list of hooks to execute.
    """
    hooks_to_execute = []
    for hook_name, hook in hooks.items():
        env_var = f"KEDRO_HOOKS_DISABLE_{hook_name.upper()}"
        if not os.getenv(env_var):
            hooks_to_execute.append(hook)

    return hooks_to_execute


def string_to_native(value: str):
    """Utility function to cast a string into it's native type."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If the string cannot be converted, return it as-is
        return value


def generate_dynamic_pipeline_mapping(
    mapping: Union[Any, Dict[str, str]], path: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Utility that we added to update the dynamic mapping through environment variables.

    It looks for env variables that start with KEDRO_DYNAMIC_PIPELINES_MAPPING, and if found update the
    corresponding entry in the pipeline mapping.

    Parameters:
        mapping: Dictionary containing the dynamic pipeline mapping.

    Returns:
        Dynamic pipeline mapping, updated with variables according to the environment.
    """

    if path is None:
        path = []

    # NOTE: We're currently not touching lists, we should unify the settings format
    # to ensure everything is specified as a dict.
    if isinstance(mapping, List):
        if path and path[-1] == "integration":
            # [-1] makes this work even if you nest this key deeper which [0] wouldn't
            if os.environ["GCP_ENV"].lower() == "dev":
                integration_sources = [item for item in mapping if not item.get("private")]
                return integration_sources
        else:
            return mapping

    if isinstance(mapping, Dict):
        result = {}
        for key, value in mapping.items():
            result[key] = generate_dynamic_pipeline_mapping(value, path=[*path, key])
        return result

    env_var = f"KEDRO_DYNAMIC_PIPELINES_MAPPING_{'_'.join(path).upper()}"
    return string_to_native(os.getenv(env_var, mapping))
