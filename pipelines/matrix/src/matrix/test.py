# %%

import mlflow
import mlflow.runs


mlflow.set_tracking_uri("http://localhost:5001")

exp = mlflow.search_experiments(filter_string="name = 'default'")
if len(exp) == 1:
    print(exp[0].experiment_id)
    experiment = exp[0].experiment_id


run = mlflow.search_runs(
    experiment_ids=[experiment],
    filter_string="run_name='abcd'",
    order_by=["start_time DESC"],
    output_format="list",
)
print(run[0].info.run_id)
# %%
import pytest

from pathlib import Path

from kedro.framework.context import KedroContext
from kedro.framework.project import configure_project, settings
from kedro.framework.hooks import _create_hook_manager

from pyspark.sql import SparkSession
from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc
from resolvers import env
from settings import merge_dicts

conf = OmegaConfigLoader(
    env="base",
    base_env="base",
    conf_source="../../conf",
    config_patterns={
        "mlflow": ["mlflow*", "mlflow*/**"],
        "spark": ["spark*", "spark*/**"],
        "globals": ["globals*", "globals*/**", "**/globals*"],
        "parameters": [
            "parameters*",
            "parameters*/**",
            "**/parameters*",
            "**/parameters*/**",
        ],
    },
    custom_resolvers={
        "merge": merge_dicts,
        "oc.env": env,
    },
)

print(conf.get("mlflow"))

# %%
