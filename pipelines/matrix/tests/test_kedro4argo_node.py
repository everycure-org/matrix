from kedro.pipeline import pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error, r2_score
from sklearn.model_selection import train_test_split

import pandas as pd
from kedro.pipeline.node import Node, node
import logging
import pytest

from matrix.kedro4argo_node import ArgoResourceConfig, ArgoNode, argo_node
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from matrix.kedro4argo_node import (
    KUBERNETES_DEFAULT_LIMIT_CPU,
    KUBERNETES_DEFAULT_LIMIT_RAM,
    KUBERNETES_DEFAULT_REQUEST_CPU,
    KUBERNETES_DEFAULT_REQUEST_RAM,
)


def dummy_func(x) -> int:
    return x


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_request", -1),
        ("cpu_limit", 0),
        ("memory_request", -2.5),
        ("memory_limit", -0.1),
    ],
)
def test_negative_resources_raise_error(field, value):
    """Test that negative or zero resource values raise ValueError."""
    with pytest.raises(ValueError, match="Resource values must be positive"):
        ArgoResourceConfig(**{field: value})


@pytest.mark.parametrize(
    "cpu_request, cpu_limit, memory_request, memory_limit, match",
    [
        (2.0, 1.0, None, None, "CPU limit must be greater than or equal to CPU request"),
        (None, None, 4.0, 2.0, "Memory limit must be greater than or equal to memory request"),
    ],
)
def test_invalid_resource_constraints(cpu_request, cpu_limit, memory_request, memory_limit, match):
    """Test that invalid resource constraints raise ValueError."""
    kwargs = {}
    if cpu_request is not None:
        kwargs["cpu_request"] = cpu_request
    if cpu_limit is not None:
        kwargs["cpu_limit"] = cpu_limit
    if memory_request is not None:
        kwargs["memory_request"] = memory_request
    if memory_limit is not None:
        kwargs["memory_limit"] = memory_limit

    with pytest.raises(ValueError, match=match):
        ArgoResourceConfig(**kwargs)


@pytest.mark.parametrize(
    "cpu_limit, memory_limit",
    [
        (100, 1000),
        (200, 2000),
    ],
)
def test_high_resource_values_warning(cpu_limit, memory_limit):
    """Test that unrealistically high resource values trigger a warning."""
    with pytest.warns(UserWarning, match="CPU .* and memory .* limits and requests are unrealistically high"):
        ArgoResourceConfig(
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )


@pytest.mark.parametrize(
    "cpu_request, cpu_limit, memory_request, memory_limit",
    [
        (2.0, 2.0, 4.0, 4.0),  # Equal limits and requests
        (1.0, 2.0, 2.0, 4.0),  # Limits higher than requests
    ],
)
def test_valid_resource_configuration(cpu_request, cpu_limit, memory_request, memory_limit):
    """Test valid resource configuration scenarios."""
    config = ArgoResourceConfig(
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        memory_request=memory_request,
        memory_limit=memory_limit,
    )
    assert config.cpu_request == cpu_request
    assert config.cpu_limit == cpu_limit
    assert config.memory_request == memory_request
    assert config.memory_limit == memory_limit

    config = ArgoResourceConfig(num_gpus=1)
    assert config.num_gpus == 1

    config = ArgoResourceConfig(num_gpus=0)
    assert config.num_gpus == 0


@pytest.mark.parametrize(
    "cpu_request, cpu_limit, memory_request, memory_limit",
    [
        (0.25, 1, 0.25, 1),  # Equal limits and requests
        (1.0, 2.0, 0.25, 0.5),  # Limits higher than requests
    ],
)
def test_fractional_resources_not_accepted(cpu_request, cpu_limit, memory_request, memory_limit):
    """Test that fractional resources are not accepted."""
    with pytest.raises(ValueError, match="Currently fractional resource values are not accepted"):
        ArgoResourceConfig(
            cpu_request=cpu_request,
            cpu_limit=cpu_limit,
            memory_request=memory_request,
            memory_limit=memory_limit,
        )


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            {"cpu_request": 1, "cpu_limit": 2, "memory_request": 16, "memory_limit": 32, "num_gpus": 0},
            {"cpu_request": 1, "cpu_limit": 2, "memory_request": "16Gi", "memory_limit": "32Gi", "num_gpus": 0},
        ),
        (
            {"cpu_request": 2, "cpu_limit": 4, "memory_request": 32, "memory_limit": 64, "num_gpus": 0},
            {"cpu_request": 2, "cpu_limit": 4, "memory_request": "32Gi", "memory_limit": "64Gi", "num_gpus": 0},
        ),
        (
            {"cpu_request": 1, "cpu_limit": 16, "memory_request": 64, "memory_limit": 128, "num_gpus": 1},
            {"cpu_request": 1, "cpu_limit": 16, "memory_request": "64Gi", "memory_limit": "128Gi", "num_gpus": 1},
        ),
    ],
)
def test_serialization(values, expected):
    config = ArgoResourceConfig(
        cpu_request=values["cpu_request"],
        cpu_limit=values["cpu_limit"],
        memory_request=values["memory_request"],
        memory_limit=values["memory_limit"],
        num_gpus=values["num_gpus"],
    )
    assert config.model_dump() == expected


def get_parametrized_node(node_class: Node) -> Node:
    def dummy_func(x: int) -> int:
        return 2 * x

    return node_class(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs="int_number_ds_out",
        name="dummy_node",
        tags=["dummy_tag"],
        namespace="dummy_namespace",
    )


def test_parametrized_node():
    normal_node = get_parametrized_node(Node)
    assert normal_node.func(2) == 4

    k8s_node = get_parametrized_node(ArgoNode)
    assert k8s_node.func(2) == 4


def test_argo_node_default_config():
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert k8s_node.argo_config.num_gpus == 0


def test_argo_node_can_request_gpu():
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        argo_config=ArgoResourceConfig(num_gpus=1),
    )
    assert k8s_node.argo_config.num_gpus == 1


def test_validate_values_are_sane():
    """Test that validate_values_are_sane raises warnings for unrealistic values."""
    with pytest.warns(UserWarning, match="CPU .* and memory .* limits and requests are unrealistically high"):
        ArgoResourceConfig(cpu_limit=100, memory_limit=1000)


def get_parallel_pipelines() -> Pipeline:
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
                argo_config=ArgoResourceConfig(
                    cpu_request=2,
                    cpu_limit=4,
                    memory_request=32,
                    memory_limit=128,
                    num_gpus=1,
                ),
                tags=["k8s_pipeline"],
            ),
            ArgoNode(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
                argo_config=ArgoResourceConfig(
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


def test_parallel_pipelines(caplog):
    k8s_pipeline, standard_pipeline = get_parallel_pipelines()

    assert k8s_pipeline.nodes[0].tags == {"k8s_pipeline"}
    assert standard_pipeline.nodes[0].tags == {"standard_pipeline"}

    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "model_input_table@pandas": pd.DataFrame({"price": [100, 200, 300, 400]}),
            "params:model_options": {"features": ["price"], "test_size": 0.25, "random_state": 42},
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(k8s_pipeline, catalog)
    assert successful_run_msg in caplog.text

    caplog.clear()

    SequentialRunner().run(standard_pipeline, catalog)
    assert successful_run_msg in caplog.text

    assert all(isinstance(node, ArgoNode) for node in k8s_pipeline.nodes)
    assert all(isinstance(node, Node) for node in standard_pipeline.nodes)


def test_argo_node_factory():
    argo_node_instance = argo_node(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert argo_node_instance.argo_config.cpu_request == KUBERNETES_DEFAULT_REQUEST_CPU
    assert argo_node_instance.argo_config.cpu_limit == KUBERNETES_DEFAULT_LIMIT_CPU
    assert argo_node_instance.argo_config.memory_request == KUBERNETES_DEFAULT_REQUEST_RAM
    assert argo_node_instance.argo_config.memory_limit == KUBERNETES_DEFAULT_LIMIT_RAM
    assert argo_node_instance.argo_config.num_gpus == 0

    kedro_node = node(func=dummy_func, inputs=["int_number_ds_in"], outputs=["int_number_ds_out"])
    assert argo_node_instance.func == kedro_node.func
    assert argo_node_instance.inputs == kedro_node.inputs
    assert argo_node_instance.outputs == kedro_node.outputs


def test_fuse_config() -> None:
    argo_config = ArgoResourceConfig(
        cpu_request=1,
        cpu_limit=2,
        memory_request=16,
        memory_limit=32,
        num_gpus=1,
    )
    other_argo_config = ArgoResourceConfig(
        cpu_request=3,
        cpu_limit=4,
        memory_request=32,
        memory_limit=64,
        num_gpus=0,
    )
    argo_config.fuse_config(other_argo_config)
    assert argo_config.cpu_request == 3
    assert argo_config.cpu_limit == 4
    assert argo_config.memory_request == 32
    assert argo_config.memory_limit == 64
    assert argo_config.num_gpus == 1


def test_initialization_of_pipeline_with_k8s_nodes():
    nodes = [
        ArgoNode(
            func=dummy_func,
            inputs=["int_number_ds_in"],
            outputs=["int_number_ds_out"],
            argo_config=ArgoResourceConfig(
                cpu_request=1,
                cpu_limit=2,
                memory_request=16,
                memory_limit=32,
                num_gpus=1,
            ),
        ),
        ArgoNode(
            func=dummy_func,
            inputs=["int_number_ds_out"],
            outputs=["int_number_ds_out_2"],
            argo_config=ArgoResourceConfig(
                cpu_request=1,
                cpu_limit=2,
                memory_request=16,
                memory_limit=32,
                num_gpus=1,
            ),
        ),
    ]

    k8s_pipeline_without_tags = Pipeline(
        nodes=nodes,
    )

    assert isinstance(k8s_pipeline_without_tags.nodes[0], ArgoNode)
    assert isinstance(k8s_pipeline_without_tags.nodes[1], ArgoNode)

    k8s_pipeline_without_tags_from_function = pipeline(nodes)

    assert isinstance(k8s_pipeline_without_tags_from_function.nodes[0], ArgoNode)
    assert isinstance(k8s_pipeline_without_tags_from_function.nodes[1], ArgoNode)

    k8s_pipeline_with_tags = Pipeline(
        nodes=nodes,
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    assert isinstance(k8s_pipeline_with_tags.nodes[0], ArgoNode)
    assert isinstance(k8s_pipeline_with_tags.nodes[1], ArgoNode)

    k8s_pipeline_with_tags_from_function = pipeline(nodes, tags=["argowf.fuse", "argowf.fuse-group.dummy"])

    assert isinstance(k8s_pipeline_with_tags_from_function.nodes[0], ArgoNode)
    assert isinstance(k8s_pipeline_with_tags_from_function.nodes[1], ArgoNode)


def test_copy_k8s_node():
    argo_node = ArgoNode(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        argo_config=ArgoResourceConfig(
            cpu_request=1,
            cpu_limit=2,
            memory_request=16,
            memory_limit=32,
            num_gpus=1,
        ),
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )
    copied_k8s_node = argo_node._copy()
    assert copied_k8s_node.argo_config.cpu_request == 1
    assert copied_k8s_node.argo_config.cpu_limit == 2
    assert copied_k8s_node.argo_config.memory_request == 16
    assert copied_k8s_node.argo_config.memory_limit == 32
    assert copied_k8s_node.argo_config.num_gpus == 1
    assert copied_k8s_node.tags == {"argowf.fuse", "argowf.fuse-group.dummy"}

    overwritten_k8s_node = argo_node._copy(
        argo_config=ArgoResourceConfig(cpu_request=3, cpu_limit=4, memory_request=32, memory_limit=64, num_gpus=0)
    )
    assert overwritten_k8s_node.argo_config.cpu_request == 3
    assert overwritten_k8s_node.argo_config.cpu_limit == 4
    assert overwritten_k8s_node.argo_config.memory_request == 32
    assert overwritten_k8s_node.argo_config.memory_limit == 64
    assert overwritten_k8s_node.argo_config.num_gpus == 0
