import os
from pathlib import Path
from typing import Generator

import pyspark.sql as ps
import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings
from matrix.resolvers import cast_to_int, if_null, merge_dicts
from matrix.settings import _load_setting
from omegaconf.resolvers import oc


@pytest.fixture(scope="session")
def matrix_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_resources_root() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def conf_source(matrix_root: Path) -> Path:
    return matrix_root / settings.CONF_SOURCE


def build_config_loader(env: str, conf_source: Path) -> OmegaConfigLoader:
    """Instantiate a config loader."""
    return OmegaConfigLoader(
        env=env,
        base_env="base",
        default_run_env="base",
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
        custom_resolvers={
            "merge": merge_dicts,
            "oc.env": oc.env,
            "setting": _load_setting,
            "oc.int": cast_to_int,
            "if_null": if_null,
        },
    )


def build_kedro_context(config_loader: OmegaConfigLoader) -> KedroContext:
    """Instantiate a KedroContext."""
    return KedroContext(
        env="cloud",
        package_name="matrix",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


@pytest.fixture(scope="session")
def cloud_config_loader(conf_source: Path) -> OmegaConfigLoader:
    return build_config_loader("cloud", conf_source)


@pytest.fixture(scope="session")
def base_config_loader(conf_source: Path) -> OmegaConfigLoader:
    return build_config_loader("base", conf_source)


@pytest.fixture(scope="session")
def cloud_kedro_context(conf_source: Path) -> KedroContext:
    config_loader = build_config_loader("cloud", conf_source)
    return build_kedro_context(config_loader)


@pytest.fixture(scope="session")
def base_kedro_context(conf_source: Path) -> KedroContext:
    config_loader = build_config_loader("base", conf_source)
    return build_kedro_context(config_loader)


@pytest.fixture(scope="session")
def spark() -> Generator[ps.SparkSession, None, None]:
    """Instantiate the Spark session."""
    spark = (
        ps.SparkSession.builder.config("spark.sql.shuffle.partitions", 1)
        .config("spark.executorEnv.PYTHONPATH", "src")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # For the spark session tests that don't involve `OmegaConfig`, which in turns uses `spark.yml`
        # that takes care of adding the unit (`g`), we make sure that the raw env var has the correct unit added.
        .config("spark.driver.memory", f"{os.environ['SPARK_DRIVER_MEMORY']}g")
        .master("local")
        .appName("tests")
        .getOrCreate()
    )
    yield spark
