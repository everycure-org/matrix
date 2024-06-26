"""Module with pytest fixtures."""
import pytest

from pathlib import Path

from kedro.framework.context import KedroContext
from kedro.framework.project import configure_project, settings
from kedro.framework.hooks import _create_hook_manager

from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc
from matrix.resolvers import merge_dicts


@pytest.fixture(name="conf_source", scope="session")
def conf_source_fixture() -> str:
    """Return the project path."""
    return str(Path.cwd() / settings.CONF_SOURCE)


@pytest.fixture(name="config_loader", scope="session")
def config_loader_fixture(conf_source) -> OmegaConfigLoader:
    """Instantiate a config loader."""
    return OmegaConfigLoader(
        env="base",
        base_env="base",
        conf_source=conf_source,
        config_patterns={
            "spark": ["spark*", "spark*/**"],
            "globals": ["globals*", "globals*/**", "**/globals*"],
            "parameters": [
                "parameters*",
                "parameters*/**",
                "**/parameters*",
                "**/parameters*/**",
            ],
        },
        custom_resolvers={"merge": merge_dicts, "oc.env": oc.env},
    )


@pytest.fixture(name="kedro_context", scope="session")
def kedro_context_fixture(config_loader) -> KedroContext:
    """Instantiate a KedroContext."""
    return KedroContext(
        env="base",
        package_name="matrix",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


@pytest.fixture(name="configure_matrix_project", scope="session")
def configure_matrix_project_fixture():
    """Configure the project for testing."""
    configure_project("matrix")
