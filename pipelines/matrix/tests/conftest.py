from typing import Generator
import pytest

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.project import settings
from kedro.framework.hooks import _create_hook_manager

from pyspark.sql import SparkSession
from omegaconf.resolvers import oc
from matrix.resolvers import merge_dicts


@pytest.fixture(scope="session")
def matrix_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_resources_root() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def conf_source(matrix_root: Path) -> Path:
    return matrix_root / settings.CONF_SOURCE


@pytest.fixture(scope="session")
def config_loader(conf_source: Path) -> OmegaConfigLoader:
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


@pytest.fixture(scope="session")
def kedro_context(config_loader: OmegaConfigLoader) -> KedroContext:
    """Instantiate a KedroContext."""
    return KedroContext(
        env="base",
        package_name="matrix",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Instantiate the Spark session."""
    spark = (
        SparkSession.builder.config("spark.sql.shuffle.partitions", 1)
        .config("spark.executorEnv.PYTHONPATH", "src")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .master("local")
        .appName("tests")
        .getOrCreate()
    )
    yield spark
