from dotenv import load_dotenv
from everycure.datasets.kedro.hooks import GitStorageHook
from kedro.config import OmegaConfigLoader

from .resolvers import env

load_dotenv()

HOOKS = (GitStorageHook(repo_url="https://github.com/everycure-org/datasets", branch="main"),)

CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        "oc.env": env,
    },
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
