"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

import sys
from pathlib import Path

from kedro.config import OmegaConfigLoader  # noqa: E402

# Instantiated project hooks.
from review_list.hooks import SparkHooks  # noqa: E402

# Add the matrix pipeline source to the Python path for local imports
matrix_src_path = Path(__file__).parent.parent.parent.parent / "matrix" / "src"
if matrix_src_path.exists():
    sys.path.insert(0, str(matrix_src_path))

from matrix.resolvers import cast_to_int, env, if_null, merge_dicts  # noqa: E402

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (SparkHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.


CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft", "globals": "soft"},
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
        "mlflow": ["mlflow*", "mlflow*/**"],
        "globals": ["globals*", "globals*/**", "**/globals*"],
        "parameters": [
            "parameters*",
            "parameters*/**",
            "**/parameters*",
            "**/parameters*/**",
        ],
    },
    "custom_resolvers": {
        "merge": merge_dicts,
        "oc.env": env,
        "oc.int": cast_to_int,
        # "setting": _load_setting,
        "if_null": if_null,
    },
}


# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
