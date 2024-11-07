from kedro.pipeline import pipeline, Pipeline
from kedro.pipeline.node import node

import pandas as pd
from kedro.pipeline.node import Node
import logging
import pytest

from matrix.kedro_extension import KubernetesExecutionConfig, ArgoNode, argo_node
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from matrix.settings import (
    KUBERNETES_DEFAULT_LIMIT_CPU,
    KUBERNETES_DEFAULT_LIMIT_RAM,
    KUBERNETES_DEFAULT_REQUEST_CPU,
    KUBERNETES_DEFAULT_REQUEST_RAM,
)


def dummy_func(x) -> int:
    return x


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


def test_default_values_in_k8s_config_matches_settings():
    """Test that default values in KubernetesExecutionConfig match settings."""
    config = KubernetesExecutionConfig()

    assert config.cpu_request == KUBERNETES_DEFAULT_REQUEST_CPU
    assert config.cpu_limit == KUBERNETES_DEFAULT_LIMIT_CPU
    assert config.memory_request == KUBERNETES_DEFAULT_REQUEST_RAM
    assert config.memory_limit == KUBERNETES_DEFAULT_LIMIT_RAM
    assert not config.use_gpu  # Default should be False


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


@pytest.mark.parametrize("node_class", [Node, ArgoNode])
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
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert not k8s_node.k8s_config.use_gpu


def test_kubernetes_node_can_request_gpu():
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        k8s_config=KubernetesExecutionConfig(use_gpu=True),
    )
    assert k8s_node.k8s_config.use_gpu


def test_validate_values_are_sane():
    """Test that validate_values_are_sane raises warnings for unrealistic values."""
    with pytest.warns(UserWarning, match="CPU .* and memory .* limits and requests are unrealistically high"):
        KubernetesExecutionConfig(cpu_limit=100, memory_limit=1000)


def test_default_values_in_k8s_node_config_match_settings():
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert not k8s_node.k8s_config.use_gpu
    assert k8s_node.k8s_config.cpu_request == KUBERNETES_DEFAULT_REQUEST_CPU
    assert k8s_node.k8s_config.cpu_limit == KUBERNETES_DEFAULT_LIMIT_CPU
    assert k8s_node.k8s_config.memory_request == KUBERNETES_DEFAULT_REQUEST_RAM
    assert k8s_node.k8s_config.memory_limit == KUBERNETES_DEFAULT_LIMIT_RAM


def test_k8s_node_can_override_default_values():
    k8s_node = ArgoNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        k8s_config=KubernetesExecutionConfig(
            cpu_request=1,
            cpu_limit=2,
            memory_request=16,
            memory_limit=32,
        ),
    )
    assert k8s_node.k8s_config.cpu_request == 1
    assert k8s_node.k8s_config.cpu_limit == 2
    assert k8s_node.k8s_config.memory_request == 16
    assert k8s_node.k8s_config.memory_limit == 32


def test_parallel_pipelines(caplog, parallel_pipelines):
    k8s_pipeline, standard_pipeline = parallel_pipelines

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


def test_kubernetes_node_factory():
    k8s_node = argo_node(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert k8s_node.k8s_config.cpu_request == KUBERNETES_DEFAULT_REQUEST_CPU
    assert k8s_node.k8s_config.cpu_limit == KUBERNETES_DEFAULT_LIMIT_CPU
    assert k8s_node.k8s_config.memory_request == KUBERNETES_DEFAULT_REQUEST_RAM
    assert k8s_node.k8s_config.memory_limit == KUBERNETES_DEFAULT_LIMIT_RAM
    assert not k8s_node.k8s_config.use_gpu

    kedro_node = node(func=dummy_func, inputs=["int_number_ds_in"], outputs=["int_number_ds_out"])
    assert k8s_node.func == kedro_node.func
    assert k8s_node.inputs == kedro_node.inputs
    assert k8s_node.outputs == kedro_node.outputs


def test_fuse_config() -> None:
    k8s_config = KubernetesExecutionConfig(cpu_request=1, cpu_limit=2, memory_request=16, memory_limit=32, use_gpu=True)
    other_k8s_config = KubernetesExecutionConfig(
        cpu_request=3, cpu_limit=4, memory_request=32, memory_limit=64, use_gpu=False
    )
    k8s_config.fuse_config(other_k8s_config)
    assert k8s_config.cpu_request == 3
    assert k8s_config.cpu_limit == 4
    assert k8s_config.memory_request == 32
    assert k8s_config.memory_limit == 64
    assert k8s_config.use_gpu


def test_k8s_pipeline_with_fused_nodes(parallel_pipelines):
    k8s_pipeline, standard_pipeline = parallel_pipelines
    assert all(isinstance(node, ArgoNode) for node in k8s_pipeline.nodes)


def test_initialization_of_pipeline_with_k8s_nodes():
    nodes = [
        ArgoNode(
            func=dummy_func,
            inputs=["int_number_ds_in"],
            outputs=["int_number_ds_out"],
            k8s_config=KubernetesExecutionConfig(
                cpu_request=1,
                cpu_limit=2,
                memory_request=16,
                memory_limit=32,
                use_gpu=True,
            ),
        ),
        ArgoNode(
            func=dummy_func,
            inputs=["int_number_ds_out"],
            outputs=["int_number_ds_out_2"],
            k8s_config=KubernetesExecutionConfig(
                cpu_request=1,
                cpu_limit=2,
                memory_request=16,
                memory_limit=32,
                use_gpu=True,
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
    k8s_node = ArgoNode(
        func=dummy_func,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
        k8s_config=KubernetesExecutionConfig(
            cpu_request=1,
            cpu_limit=2,
            memory_request=16,
            memory_limit=32,
            use_gpu=True,
        ),
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )
    copied_k8s_node = k8s_node._copy()
    assert copied_k8s_node.k8s_config.cpu_request == 1
    assert copied_k8s_node.k8s_config.cpu_limit == 2
    assert copied_k8s_node.k8s_config.memory_request == 16
    assert copied_k8s_node.k8s_config.memory_limit == 32
    assert copied_k8s_node.k8s_config.use_gpu
    assert copied_k8s_node.tags == {"argowf.fuse", "argowf.fuse-group.dummy"}

    overwritten_k8s_node = k8s_node._copy(
        k8s_config=KubernetesExecutionConfig(
            cpu_request=3, cpu_limit=4, memory_request=32, memory_limit=64, use_gpu=False
        )
    )
    assert overwritten_k8s_node.k8s_config.cpu_request == 3
    assert overwritten_k8s_node.k8s_config.cpu_limit == 4
    assert overwritten_k8s_node.k8s_config.memory_request == 32
    assert overwritten_k8s_node.k8s_config.memory_limit == 64
    assert not overwritten_k8s_node.k8s_config.use_gpu
