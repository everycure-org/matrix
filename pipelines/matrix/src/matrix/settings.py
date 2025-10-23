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
from matrix.utils.hook_utilities import (
    determine_hooks_to_execute,
    disable_private_datasets,
    generate_dynamic_pipeline_mapping,
)

from .resolvers import cast_to_int, env, if_null, merge_dicts

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

# https://getindata.com/blog/kedro-dynamic-pipelines/

# Using lambda to delay the evaluation until the INCLUDE_PRIVATE_DATASETS env var is set, parsed from a cli option.
DYNAMIC_PIPELINES_MAPPING = lambda: disable_private_datasets(
    generate_dynamic_pipeline_mapping(
        {
            "cross_validation": {
                "n_cross_val_folds": 5,
            },
            "num_shards": 3,
            "integration": [
                # {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False},
                # {"name": "spoke", "integrate_in_kg": True, "is_private": True},
                # {"name": "embiology", "integrate_in_kg": True, "is_private": True},
                # {"name": "robokop", "integrate_in_kg": True, "is_private": False},
                {"name": "primekg", "integrate_in_kg": True, "is_private": False, "validate": True},
                {"name": "drug_list", "integrate_in_kg": False, "has_edges": False, "is_core": True},
                {"name": "disease_list", "integrate_in_kg": False, "has_edges": False, "is_core": True},
                {
                    "name": "kgml_xdtd_ground_truth",
                    "has_nodes": False,
                    "has_edges": True,
                    "integrate_in_kg": False,
                },
                # {
                #     "name": "ec_ground_truth",
                #     "has_nodes": False,
                #     "has_edges": True,
                #     "integrate_in_kg": False,
                # },
                {
                    "name": "drugbank_ground_truth",
                    "has_nodes": False,
                    "has_edges": True,
                    "integrate_in_kg": False,
                    "is_private": True,
                },
                # {"name": "drugmech", "integrate_in_kg": False, "has_nodes": False},
                # {"name": "ec_clinical_trails", "integrate_in_kg": False},
                # {"name": "off_label", "integrate_in_kg": False, "has_nodes": False},
                # TODO: enable orchard once permissions are clarified
                {"name": "orchard", "integrate_in_kg": False, "has_nodes": False, "is_private": True},
            ],
            # model_name suggestions: xg_baseline, xg_ensemble, rf, xg_synth, lightGBM
            "modelling": [
                # {
                #     "model_name": "xg_ensemble",
                # },
                {
                    "model_name": "xg_synth",
                },
            ],
            "evaluation": [
                {"evaluation_name": "simple_classification"},
                {"evaluation_name": "disease_specific"},
                {"evaluation_name": "full_matrix_negatives"},
                {"evaluation_name": "full_matrix"},
                {"evaluation_name": "simple_classification_trials"},
                {"evaluation_name": "disease_specific_trials"},
                {"evaluation_name": "full_matrix_trials"},
                {"evaluation_name": "disease_specific_off_label"},
                {"evaluation_name": "full_matrix_off_label"},
            ],
            "stability": [
                {"stability_name": "stability_overlap"},
                {"stability_name": "stability_ranking"},
                {
                    "stability_name": "rank_commonality"
                },  # note - rank_commonality will be only used if you have a shared commonality@k and spearman@k metrics
            ],
            "document_kg": [
                {"name": "infores", "source_type": "external_registry"},
                {"name": "reusabledata", "source_type": "external_registry", "has_mapping": True},
                {"name": "kgregistry", "source_type": "external_registry", "has_mapping": True},
                {"name": "matrix_curated", "source_type": "matrix_curated"},
                {"name": "matrix_reviews", "source_type": "matrix_curated"},
            ],
        }
    )
)


def _load_setting(path):
    """Utility function to load a settings value from the data catalog."""
    path = path.split(".")
    obj = DYNAMIC_PIPELINES_MAPPING()
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
    "custom_resolvers": {
        "merge": merge_dicts,
        "oc.env": env,
        "oc.int": cast_to_int,
        "setting": _load_setting,
        "if_null": if_null,
    },
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
