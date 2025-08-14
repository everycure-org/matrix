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
    mapping: Union[Any, Dict[str, str]],
    path: Optional[List[str]] = None,
    integrate_in_kg: bool = True,
    is_private: bool = False,
    has_edges: bool = True,
    has_nodes: bool = True,
    is_core: bool = False,
) -> Dict[str, Any]:
    """Utility that we added to update the dynamic mapping through environment variables.

    It looks for env variables that start with KEDRO_DYNAMIC_PIPELINES_MAPPING, and if found update the
    corresponding entry in the pipeline mapping. Also applies integration defaults to integration entries.

    Parameters:
        mapping: Dictionary containing the dynamic pipeline mapping.
        path: Current path in the mapping hierarchy.
        integrate_in_kg: Default value for integrate_in_kg in integration entries.
        is_private: Default value for is_private in integration entries.
        has_edges: Default value for has_edges in integration entries.
        has_nodes: Default value for has_nodes in integration entries.
        is_core: Default value for is_core in integration entries.

    Returns:
        Dynamic pipeline mapping, updated with variables according to the environment.
    """

    if path is None:
        path = []

    # Create integration defaults dict from function arguments
    integration_defaults = {
        "integrate_in_kg": integrate_in_kg,
        "is_private": is_private,
        "has_edges": has_edges,
        "has_nodes": has_nodes,
        "is_core": is_core,
    }

    # NOTE: We're currently not touching lists, we should unify the settings format
    # to ensure everything is specified as a dict.
    if isinstance(mapping, List):
        # Apply defaults to integration entries if we're in the integration section
        if len(path) > 0 and path[-1] == "integration":
            result = []
            for item in mapping:
                if isinstance(item, Dict) and "name" in item:
                    # Apply defaults, but let existing values override
                    updated_item = {**integration_defaults, **item}
                    result.append(updated_item)
                else:
                    result.append(item)
            return result
        return mapping

    if isinstance(mapping, Dict):
        result = {}
        for key, value in mapping.items():
            result[key] = generate_dynamic_pipeline_mapping(
                value,
                path=[*path, key],
                integrate_in_kg=integrate_in_kg,
                is_private=is_private,
                has_edges=has_edges,
                has_nodes=has_nodes,
                is_core=is_core,
            )
        return result

    env_var = f"KEDRO_DYNAMIC_PIPELINES_MAPPING_{'_'.join(path).upper()}"
    return string_to_native(os.getenv(env_var, mapping))


def disable_private_datasets(config: dict) -> dict:
    if not os.getenv("INCLUDE_PRIVATE_DATASETS", "") == "1":
        config["integration"] = [item for item in config["integration"] if not item.get("is_private")]
    return config
