"""Project settings.

There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html.
"""

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from pandas_viz.hooks import ProjectHooks
# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader  # noqa: E402
from kedro_mlflow.framework.hooks import MlflowHook

import matrix.hooks as hooks

from .resolvers import env, merge_dicts

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (
    hooks.NodeTimerHooks(),
    MlflowHook(),
    hooks.MLFlowHooks(),
    hooks.SparkHooks(),
)

# Installed plugins for which to disable hook auto-registration.
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow",)

# Class that manages storing KedroSession data.
from pathlib import Path  # noqa: E402

from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore  # noqa: E402

SESSION_STORE_CLASS = SQLiteStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2])}

# Directory that holds configuration.
CONF_SOURCE = "conf"


CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft", "mlflow": "soft", "globals": "soft"},
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
    },
}

# https://getindata.com/blog/kedro-dynamic-pipelines/
DYNAMIC_PIPELINES_MAPPING = {
    "modelling": [
        {"model_name": "xg_baseline", "num_shards": 1, "run_inference": False},
        {"model_name": "xg_ensemble", "num_shards": 3, "run_inference": True},
        {"model_name": "rf", "num_shards": 1, "run_inference": False},
        {"model_name": "xg_synth", "num_shards": 1, "run_inference": False},
    ],
    "evaluation": [
        {"evaluation_name": "simple_ground_truth_classification"},
        {"evaluation_name": "continuous_ground_truth_classification"},
        {"evaluation_name": "disease_centric_matrix"},
        {"evaluation_name": "disease_specific_ranking"},
        # {"evaluation_name": "recall_at_n"},
        {"evaluation_name": "simple_ground_truth_classification_time_split"},
        {"evaluation_name": "continuous_ground_truth_classification_time_split"},
        {"evaluation_name": "disease_centric_matrix_time_split"},
        {"evaluation_name": "disease_specific_ranking_time_split"},
    ],
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
