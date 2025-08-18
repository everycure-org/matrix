from typing import Dict, List, Tuple

import pytest
import yaml
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matrix.argo import FusedNode, clean_name, fuse, generate_argo_config, get_dependencies
from matrix.kedro4argo_node import (
    KUBERNETES_DEFAULT_LIMIT_CPU,
    KUBERNETES_DEFAULT_LIMIT_RAM,
    KUBERNETES_DEFAULT_NUM_GPUS,
    KUBERNETES_DEFAULT_REQUEST_CPU,
    KUBERNETES_DEFAULT_REQUEST_RAM,
    ArgoNode,
    ArgoResourceConfig,
)


def dummy_fn(*args):
    return "dummy"


@pytest.fixture()
def nodes_where_first_is_input_for_second():
    nodes = [
        ArgoNode(
            func=dummy_fn,
            inputs=["dataset_a", "dataset_b"],
            outputs="dataset_1@pandas",
            argo_config=ArgoResourceConfig(
                cpu_request=1,
                cpu_limit=2,
                memory_request=16,
                memory_limit=32,
                num_gpus=1,
            ),
        ),
        ArgoNode(
            func=dummy_fn,
            inputs=[
                "dataset_1@spark",
            ],
            outputs="dataset_2",
            argo_config=ArgoResourceConfig(
                cpu_request=2,
                cpu_limit=2,
                memory_request=32,
                memory_limit=64,
                num_gpus=0,
            ),
        ),
    ]
    return nodes


@pytest.fixture()
def simple_node():
    return Node(func=dummy_fn, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")


@pytest.fixture()
def fused_node():
    return FusedNode(depth=0)


@pytest.mark.parametrize("node_class", [Node, ArgoNode])
def test_no_nodes_fused_when_no_fuse_options(node_class):
    pipeline_with_no_fusing_options = Pipeline(
        nodes=[
            node_class(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="first",
            ),
            node_class(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],  # inputs are different than outputs of previous node
                outputs="dataset_3",
                name="second",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_with_no_fusing_options)

    assert len(fused) == len(
        pipeline_with_no_fusing_options.nodes
    ), "No nodes should be fused when no fuse options are provided"


@pytest.mark.parametrize("node_class", [Node, ArgoNode])
def test_simple_fusing(node_class):
    pipeline_where_first_node_is_input_for_second = Pipeline(
        nodes=[
            node_class(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1@pandas",
            ),
            node_class(
                func=dummy_fn,
                inputs=[
                    "dataset_1@spark",
                ],
                outputs="dataset_2",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_where_first_node_is_input_for_second)

    assert len(fused) == 1, "Only one node should be fused"
    assert fused[0].name == "dummy", "Fused node should have name 'dummy'"
    assert fused[0].outputs == set(
        ["dataset_1", "dataset_2"]
    ), "Fused node should have outputs 'dataset_1' and 'dataset_2'"
    assert len(fused[0]._parents) == 0, "Fused node should have no parents"


@pytest.mark.parametrize("node_class", [Node, ArgoNode])
def test_no_multiple_parents_no_fusing(node_class):
    pipeline_one2many_fusing_possible = Pipeline(
        nodes=[
            node_class(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1",
                name="first_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],
                outputs="dataset_3",
                name="child_node",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_one2many_fusing_possible)

    assert len(fused) == len(
        pipeline_one2many_fusing_possible.nodes
    ), "No fusing has been performed, as child node can be fused to different parents."


@pytest.mark.parametrize("node_class", [Node, ArgoNode])
def test_fusing_multiple_parents(node_class):
    pipeline_with_multiple_parents = Pipeline(
        nodes=[
            node_class(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs=["dataset_1"],
                name="first_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=None,
                outputs="dataset_3",
                name="third_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=[
                    "dataset_1",
                    "dataset_2",
                ],
                outputs="dataset_4",
                name="child_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=["dataset_3", "dataset_4"],
                outputs="dataset_5",
                name="grandchild_node",
            ),
            node_class(
                func=dummy_fn,
                inputs=["dataset_5"],
                outputs="dataset_6",
                name="grandgrandchild_node",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused = fuse(pipeline_with_multiple_parents)

    assert len(fused) == 4, "Fusing of child and grandchild node, ensure correct naming"
    assert fused[3].name == "dummy", "Fused node should have name 'dummy'"
    assert (
        fused[3].nodes == "child_node,grandchild_node,grandgrandchild_node"
    ), "Fused node should have nodes 'child_node,grandchild_node,grandgrandchild_node'"
    assert fused[3].outputs == set(
        ["dataset_4", "dataset_5", "dataset_6"]
    ), "Fused node should have outputs 'dataset_4', 'dataset_5' and 'dataset_6'"
    assert set([parent.name for parent in fused[3]._parents]) == set(
        ["first_node", "second_node", "third_node"]
    ), "Fused node should have parents 'first_node', 'second_node' and 'third_node'"


def test_simple_fusing_with_argo_nodes(nodes_where_first_is_input_for_second: List[ArgoNode]):
    pipeline = Pipeline(
        nodes=nodes_where_first_is_input_for_second,
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )
    fused = fuse(pipeline)

    assert len(fused) == 1
    assert fused[0].argo_config.cpu_request == 2
    assert fused[0].argo_config.cpu_limit == 2
    assert fused[0].argo_config.memory_request == 32
    assert fused[0].argo_config.memory_limit == 64
    assert fused[0].argo_config.num_gpus == 1


def test_get_dependencies_default_different_than_task(nodes_where_first_is_input_for_second: List[ArgoNode]):
    pipeline = Pipeline(
        nodes=nodes_where_first_is_input_for_second,
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    fused_pipeline = fuse(pipeline)
    deps = get_dependencies(fused_pipeline, ArgoResourceConfig())
    assert len(deps) == 1
    assert deps[0]["name"] == "dummy"
    assert deps[0]["deps"] == []
    assert (
        deps[0]["nodes"]
        == "dummy_fn([dataset_a;dataset_b]) -> [dataset_1@pandas],dummy_fn([dataset_1@spark]) -> [dataset_2]"
    )
    assert deps[0]["tags"] == {"argowf.fuse", "argowf.fuse-group.dummy"}
    assert deps[0]["resources"] == {
        "ephemeral_storage_limit": "128Gi",
        "ephemeral_storage_request": "0Gi",
        "cpu_limit": 2,
        "cpu_request": 2,
        "memory_limit": 64,
        "memory_request": 32,
        "num_gpus": 1,
        "ephemeral_storage_limit": 128,
        "ephemeral_storage_request": 0,
    }


def test_get_dependencies_default_same_than_task(nodes_where_first_is_input_for_second: List[ArgoNode]):
    pipeline = Pipeline(
        nodes=nodes_where_first_is_input_for_second,
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )
    fused_pipeline = fuse(pipeline)
    deps = get_dependencies(
        fused_pipeline, ArgoResourceConfig(cpu_request=2, cpu_limit=2, memory_request=32, memory_limit=64, num_gpus=1)
    )
    assert len(deps) == 1
    assert deps[0]["name"] == "dummy"
    assert deps[0]["deps"] == []
    assert (
        deps[0]["nodes"]
        == "dummy_fn([dataset_a;dataset_b]) -> [dataset_1@pandas],dummy_fn([dataset_1@spark]) -> [dataset_2]"
    )
    assert deps[0]["tags"] == {"argowf.fuse", "argowf.fuse-group.dummy"}
    assert "resources" in deps[0]


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("node_name_1", "node-name-1"),  # Basic case with underscore
        ("node@name!", "node-name"),  # Special characters
        ("_node_name_", "node-name"),  # Leading and trailing underscores
        ("-node_name-", "node-name"),  # Leading and trailing dashes
        ("@@node!!", "node"),  # Leading and trailing special characters
        ("", ""),  # Empty string
        ("clean-name", "clean-name"),  # Already clean name
        ("node__name---test", "node-name-test"),  # Multiple consecutive special characters
        ("name!!!node$$$name", "name-node-name"),  # Complex case with multiple special characters
    ],
)
def test_clean_name(input_name: str, expected: str) -> None:
    """Test clean_name function with various input cases."""
    assert clean_name(input_name) == expected


def test_fused_node_initialization(fused_node: FusedNode) -> None:
    assert fused_node.depth == 0
    assert fused_node._nodes == []
    assert fused_node._parents == set()
    assert fused_node._inputs == []


def test_add_node(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert len(fused_node._nodes) == 1
    assert fused_node._nodes[0] == simple_node


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_add_parents(fused_node: FusedNode):
    parent1 = FusedNode(depth=1)
    parent2 = FusedNode(depth=1)
    fused_node.add_parents([parent1, parent2])
    assert len(fused_node._parents) == 2
    assert parent1 in fused_node._parents
    assert parent2 in fused_node._parents


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_fuses_with(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test"]

    fusable_node = Node(
        func=dummy_fn,
        inputs=["dataset_c"],
        outputs="dataset_d",
        name="fusable_node",
        tags=["argowf.fuse", "argowf.fuse-group.test"],
    )

    assert fused_node.fuses_with(fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    non_fusable_node = Node(func=dummy_fn, inputs=["dataset_x"], outputs="dataset_y", name="non_fusable_node")

    assert not fused_node.fuses_with(non_fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse"]
    assert fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert not fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_fuse_group(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse-group.test_group"]
    assert fused_node.fuse_group == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_nodes_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_fn, name="second_node"))
    assert fused_node.nodes == "simple_node,second_node"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_outputs_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_fn, outputs="dataset_d"))
    assert fused_node.outputs == {"dataset_c", "dataset_d"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_tags_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["tag1", "tag2"]
    fused_node.add_node(Node(func=dummy_fn, tags=["tag2", "tag3"]))
    assert fused_node.tags == {"tag1", "tag2", "tag3"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_name_property_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test_group"]
    fused_node.add_node(Node(func=dummy_fn, name="second_node"))
    assert fused_node.name == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_name_property_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert fused_node.name == "simple_node"


def test_get_fuse_group() -> None:
    tags = ["argowf.fuse-group.test_group", "other_tag"]
    assert FusedNode.get_fuse_group(tags) == "test_group"


def test_get_fuse_group_no_group() -> None:
    tags = ["other_tag"]
    assert FusedNode.get_fuse_group(tags) is None


def test_clean_dependencies() -> None:
    elements = ["dataset_a@pandas", "params:some_param", "dataset_b"]
    cleaned = FusedNode.clean_dependencies(elements)
    assert cleaned == ["dataset_a", "dataset_b"]


def get_argo_config(argo_default_resources: ArgoResourceConfig) -> Tuple[Dict, Dict[str, Pipeline]]:
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    run_name = "test_run"
    release_version = "test_release"
    image_tag = "test_tag"
    mlflow_experiment_id = 1
    namespace = "test_namespace"
    username = "test_user"
    mlflow_url = "https://mlflow.platform.dev.everycure.org/"
    pipeline_obj = Pipeline(
        nodes=[
            ArgoNode(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="simple_node_p1_1",
                argo_config=ArgoResourceConfig(
                    num_gpus=1,
                    cpu_request=4,
                    cpu_limit=7,
                    memory_request=16,
                    memory_limit=32,
                ),
            ),
            ArgoNode(
                func=dummy_fn,
                inputs="dataset_c",
                outputs="dataset_d",
                name="simple_node_p1_2",
            ),
        ]
    )
    pipeline_obj.name = "pipeline_one"
    argo_config_yaml = generate_argo_config(
        image=image_name,
        run_name=run_name,
        release_version=release_version,
        mlflow_experiment_id=mlflow_experiment_id,
        namespace=namespace,
        username=username,
        pipeline=pipeline_obj,
        package_name="matrix",
        release_folder_name="releases",
        environment="cloud",
        default_execution_resources=argo_default_resources,
        mlflow_url=mlflow_url,
    )

    argo_config = yaml.safe_load(argo_config_yaml)
    assert isinstance(argo_config, dict), "Argo config should be a dictionary after YAML parsing"
    return argo_config, {"pipeline_one": pipeline_obj}


@pytest.mark.parametrize(
    "argo_default_resources",
    [
        ArgoResourceConfig(
            num_gpus=0,
            cpu_request=4,
            cpu_limit=16,
            memory_request=64,
            memory_limit=64,
        )
    ],
)
def test_argo_template_config_boilerplate(argo_default_resources: ArgoResourceConfig) -> None:
    """Test the boilerplate configuration of the Argo template."""
    argo_config, pipelines = get_argo_config(argo_default_resources)
    spec = argo_config["spec"]

    # Verify kedro template
    kedro_template = next(t for t in spec["templates"] if t["name"] == "kedro")

    # Verify common configurations
    assert kedro_template["metadata"]["labels"]["app"] == "kedro-argo"

    # Verify default anti-affinity for GPU nodes
    assert "nodeAffinity" in kedro_template["affinity"]
    selector = kedro_template["affinity"]["nodeAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0]
    match_expression = selector["preference"]["matchExpressions"][0]
    assert match_expression["key"] == "cloud.google.com/gke-spot"
    assert match_expression["operator"] == "Equal"
    assert match_expression["values"] == ["true"]

    # Verify resources based on GPU configuration
    assert "podSpecPatch" in kedro_template

    # Verify pipeline templates
    templates = spec["templates"]
    pipeline_names = [template["name"] for template in templates]
    # assert that the template contains our 3 expected templates
    assert ["kedro", "neo4j", "pipeline"] == pipeline_names


def test_resources_of_argo_template_config_pipelines() -> None:
    """Test the resources configuration of the Argo template."""
    argo_default_resources = ArgoResourceConfig(
        num_gpus=KUBERNETES_DEFAULT_NUM_GPUS,
        cpu_request=KUBERNETES_DEFAULT_REQUEST_CPU,
        cpu_limit=KUBERNETES_DEFAULT_LIMIT_CPU,
        memory_request=KUBERNETES_DEFAULT_REQUEST_RAM,
        memory_limit=KUBERNETES_DEFAULT_LIMIT_RAM,
    )
    argo_config, actual_pipelines = get_argo_config(argo_default_resources)
    spec = argo_config["spec"]

    # Verify pipeline templates
    templates = spec["templates"]
    pipeline_names = [template["name"] for template in templates]
    assert ["kedro", "neo4j", "pipeline"] == pipeline_names

    # Verify pipeline_one template

    pipeline_one_template = [t for t in templates if t["name"] == "pipeline"][0]
    assert "dag" in pipeline_one_template
    # there should be two tasks in the pipeline
    assert len(pipeline_one_template["dag"]["tasks"]) == len(actual_pipelines["pipeline_one"].nodes)

    # Verify first task
    task1 = pipeline_one_template["dag"]["tasks"][0]
    assert task1["name"] == "simple-node-p1-1"
    assert task1["template"] == "kedro"

    # Verify resource parameters for first task
    actual_resources_p1_1 = actual_pipelines["pipeline_one"].nodes[0].argo_config.model_dump()
    resource_params1 = {p["name"]: p["value"] for p in task1["arguments"]["parameters"]}
    assert resource_params1["num_gpus"] == actual_resources_p1_1["num_gpus"]
    assert resource_params1["memory_request"] == actual_resources_p1_1["memory_request"]
    assert resource_params1["memory_limit"] == actual_resources_p1_1["memory_limit"]
    assert resource_params1["cpu_request"] == actual_resources_p1_1["cpu_request"]
    assert resource_params1["cpu_limit"] == actual_resources_p1_1["cpu_limit"]

    # Verify second task
    task2 = pipeline_one_template["dag"]["tasks"][1]
    assert task2["name"] == "simple-node-p1-2"
    assert task2["template"] == "kedro"

    # Verify default resource parameters for second task
    resource_params2 = {p["name"]: p["value"] for p in task2["arguments"]["parameters"]}
    assert resource_params2["num_gpus"] == 0
    assert resource_params2["memory_request"] == argo_default_resources.memory_request
    assert resource_params2["memory_limit"] == argo_default_resources.memory_limit
    assert resource_params2["cpu_request"] == argo_default_resources.cpu_request
    assert resource_params2["cpu_limit"] == argo_default_resources.cpu_limit
