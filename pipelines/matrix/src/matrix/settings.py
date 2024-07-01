"""Project settings.

There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html.
"""

from typing import Dict

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from pandas_viz.hooks import ProjectHooks
from matrix.hooks import SparkHooks

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (SparkHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
from pathlib import Path  # noqa: E402
from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore  # noqa: E402

SESSION_STORE_CLASS = SQLiteStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2])}

# Directory that holds configuration.
# CONF_SOURCE = "conf"


# Class that manages how configuration is loaded.
from .resolvers import merge_dicts
from kedro.config import OmegaConfigLoader  # noqa: E402
from omegaconf.resolvers import oc

CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft"},
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
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
        "oc.env": oc.env,
    },
}

# https://getindata.com/blog/kedro-dynamic-pipelines/
DYNAMIC_PIPELINES_MAPPING = {
    "modelling": [
        {"model_name": "xgb", "num_shards": 1},
        {"model_name": "xgc", "num_shards": 3},
        {"model_name": "kgml_xdtd", "num_shards": 1},
        {"model_name": "xg_balanced", "num_shards": 1},
    ],
    "evaluation": [
        {"evaluation_name": "simple_ground_truth_classification"},
        {"evaluation_name": "continuous_ground_truth_classification"},
        {"evaluation_name": "disease_centric_matrix"},
        {"evaluation_name": "disease_specific_ranking"},
    ],
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
