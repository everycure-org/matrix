from kedro.pipeline import pipeline
from kedro.pipeline.node import Node
import logging
import pytest

from matrix.kedro_extension import KubernetesExecutionConfig, KubernetesNode
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner


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

    k8s_node = get_parametrized_node(KubernetesNode)
    assert k8s_node.func(2) == 4


@pytest.mark.parametrize("node_class", [Node, KubernetesNode])
def test_parametrized_node_in_simple_pipeline(caplog, node_class):
    node = get_parametrized_node(node_class)
    pipeline_obj = pipeline([node])
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "int_number_ds_in": 10,
            "int_number_ds_out": 20,
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(pipeline_obj, catalog)

    assert successful_run_msg in caplog.text


def test_kubernetes_node_default_config():
    k8s_node = KubernetesNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert not k8s_node.k8s_config.use_gpu


def test_kubernetes_node_can_request_gpu():
    k8s_node = KubernetesNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        k8s_config=KubernetesExecutionConfig(use_gpu=True),
    )
    assert k8s_node.k8s_config.use_gpu


def test_default_kubernetes_config():
    """Test default configuration values."""
    config = KubernetesExecutionConfig()
    assert not config.use_gpu
    assert config.cpu_request > 0
    assert config.cpu_limit > 0
    assert config.memory_request > 0
    assert config.memory_limit > 0


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
        KubernetesExecutionConfig(**{field: value})


def test_invalid_resource_constraints():
    """Test that invalid resource constraints raise ValueError."""
    # CPU limit less than request
    with pytest.raises(ValueError, match="CPU limit must be greater than or equal to CPU request"):
        KubernetesExecutionConfig(cpu_request=2.0, cpu_limit=1.0)

    # Memory limit less than request
    with pytest.raises(ValueError, match="Memory limit must be greater than or equal to memory request"):
        KubernetesExecutionConfig(memory_request=4.0, memory_limit=2.0)


def test_high_resource_values_warning():
    """Test that unrealistically high resource values trigger a warning."""
    with pytest.warns(UserWarning, match="CPU .* and memory .* limits and requests are unrealistically high"):
        KubernetesExecutionConfig(
            cpu_limit=100,
            memory_limit=1000,
        )


def test_valid_resource_configuration():
    """Test valid resource configuration scenarios."""
    # Equal limits and requests
    config = KubernetesExecutionConfig(
        cpu_request=2.0,
        cpu_limit=2.0,
        memory_request=4.0,
        memory_limit=4.0,
    )
    assert config.cpu_request == 2.0
    assert config.cpu_limit == 2.0
    assert config.memory_request == 4.0
    assert config.memory_limit == 4.0

    # Limits higher than requests
    config = KubernetesExecutionConfig(
        cpu_request=1.0,
        cpu_limit=2.0,
        memory_request=2.0,
        memory_limit=4.0,
    )
    assert config.cpu_request == 1.0
    assert config.cpu_limit == 2.0
    assert config.memory_request == 2.0
    assert config.memory_limit == 4.0


def test_gpu_flag():
    """Test GPU flag configuration."""
    config = KubernetesExecutionConfig(use_gpu=True)
    assert config.use_gpu

    config = KubernetesExecutionConfig(use_gpu=False)
    assert not config.use_gpu
