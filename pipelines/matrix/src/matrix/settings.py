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

import matrix.hooks as matrix_hooks
from matrix.utils.hook_utilities import determine_hooks_to_execute, generate_dynamic_pipeline_mapping

from .resolvers import cast_to_int, env, merge_dicts

hooks = {
    "node_timer": matrix_hooks.NodeTimerHooks(),
    "mlflow": MlflowHook(),
    "mlflow_kedro": matrix_hooks.MLFlowHooks(),
    "spark": matrix_hooks.SparkHooks(),
    "release": matrix_hooks.ReleaseInfoHooks(),
}

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = determine_hooks_to_execute(hooks)

# Installed plugins for which to disable hook auto-registration.
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow",)

# Class that manages storing KedroSession data.
from pathlib import Path  # noqa: E402

from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore  # noqa: E402

SESSION_STORE_CLASS = SQLiteStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2])}

# https://getindata.com/blog/kedro-dynamic-pipelines/
DYNAMIC_PIPELINES_MAPPING = generate_dynamic_pipeline_mapping(
    {
        "cross_validation": {
            "n_splits": 3,
        },
        "integration": [
            {"name": "rtx_kg2"},
            # {"name": "spoke"},
            {"name": "robokop"},
            {"name": "ec_medical_team"},
        ],
        "model": {
            "name": "xg_baseline",
            "config": {
                "num_shards": 1,
                "run_inference": False,
            },
            # "xg_baseline": {"num_shards": 1, "run_inference": False},
            # "xg_ensemble": {"num_shards": 3, "run_inference": True},
            # "rf": {"num_shards": 1, "run_inference": False},
            # "xg_synth": {"num_shards": 1, "run_inference": False},
        },
        "evaluation": [
            {"evaluation_name": "simple_classification"},
            {"evaluation_name": "disease_specific"},
            {"evaluation_name": "full_matrix_negatives"},
            {"evaluation_name": "full_matrix"},
            {"evaluation_name": "simple_classification_trials"},
            {"evaluation_name": "disease_specific_trials"},
            {"evaluation_name": "full_matrix_trials"},
        ],
    }
)


def _load_setting(path):
    """Utility function to load a settings value from the data catalog."""
    path = path.split(".")
    obj = DYNAMIC_PIPELINES_MAPPING
    for p in path:
        obj = obj[p]

    return obj


# Directory that holds configuration.
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
    "custom_resolvers": {"merge": merge_dicts, "oc.env": env, "oc.int": cast_to_int, "setting": _load_setting},
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
