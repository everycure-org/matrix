from kedro.pipeline import pipeline
from kedro.pipeline.node import Node
import logging
import pytest

from matrix.kedro_extension import KubernetesNode
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


def test_kubernetes_node_can_request_gpu():
    k8s_node = KubernetesNode(
        func=lambda x: x,
        inputs=["int_number_ds_in"],
        outputs=["int_number_ds_out"],
    )
    assert k8s_node is not None
