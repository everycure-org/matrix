import os
from typing import Any


def determine_hooks_to_execute(hooks: dict[str, Any]):
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
