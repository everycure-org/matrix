from typing import Tuple, Generator
from kedro.pipeline import pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error, r2_score
from sklearn.model_selection import train_test_split

import pandas as pd
from kedro.pipeline.node import Node
import logging
import pytest

from matrix.kedro_extension import KubernetesExecutionConfig, ArgoNode

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


@pytest.fixture()
def parallel_pipelines() -> Tuple[Pipeline, Pipeline]:
    def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
        """Splits data into features and targets training and test sets.

        Args:
            data: Data containing features and target.
            parameters: Parameters defined in parameters/data_science.yml.
        Returns:
            Split data.
        """
        X = data[parameters["features"]]
        y = data["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
        )
        return X_train, X_test, y_train, y_test

    def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """Trains the linear regression model.

        Args:
            X_train: Training data of independent features.
            y_train: Training data for price.

        Returns:
            Trained model.
        """
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        return regressor

    def evaluate_model(regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Calculates and logs the coefficient of determination.

        Args:
            regressor: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.
        """
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        me = max_error(y_test, y_pred)
        logger = logging.getLogger(__name__)
        logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
        return {"r2_score": score, "mae": mae, "max_error": me}

    k8s_pipeline = pipeline(
        [
            ArgoNode(
                func=split_data,
                inputs=["model_input_table@pandas", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["k8s_pipeline"],
            ),
            ArgoNode(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
                k8s_config=KubernetesExecutionConfig(
                    cpu_request=2,
                    cpu_limit=4,
                    memory_request=32,
                    memory_limit=128,
                    use_gpu=True,
                ),
                tags=["k8s_pipeline"],
            ),
            ArgoNode(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
                k8s_config=KubernetesExecutionConfig(
                    cpu_request=1,
                    cpu_limit=2,
                    memory_request=16,
                    memory_limit=32,
                ),
                tags=["k8s_pipeline"],
            ),
        ]
    )
    standard_pipeline = pipeline(
        [
            Node(
                func=split_data,
                inputs=["model_input_table@pandas", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["standard_pipeline"],
            ),
            Node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
                tags=["standard_pipeline"],
            ),
            Node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
                tags=["standard_pipeline"],
            ),
        ]
    )

    return k8s_pipeline, standard_pipeline
