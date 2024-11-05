from unittest.mock import Mock, patch
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
import pytest
import yaml

from matrix.argo import (
    clean_name,
    fuse,
    FusedNode,
    generate_argo_config,
    get_k8s_node_affinity_tags,
    get_pipeline_as_tasks,
)
from matrix.tags import ARGO_NODE_PREFIX, NodeTags, fuse_group_tag


def dummy_fn(*args):
    return "dummy"


def test_no_nodes_fused_when_no_fuse_options():
    pipeline_with_no_fusing_options = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="first",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],  # inputs are different than outputs of previous node
                outputs="dataset_3",
                name="second",
            ),
        ],
        tags=[NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("dummy")],
    )

    fused = fuse(pipeline_with_no_fusing_options)

    assert len(fused) == len(
        pipeline_with_no_fusing_options.nodes
    ), "No nodes should be fused when no fuse options are provided"


def test_simple_fusing():
    pipeline_where_first_node_is_input_for_second = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1@pandas",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_1@spark",
                ],
                outputs="dataset_2",
            ),
        ],
        tags=[NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("dummy")],
    )

    fused = fuse(pipeline_where_first_node_is_input_for_second)

    assert len(fused) == 1, "Only one node should be fused"
    assert fused[0].name == "dummy", "Fused node should have name 'dummy'"
    assert fused[0].outputs == set(
        ["dataset_1", "dataset_2"]
    ), "Fused node should have outputs 'dataset_1' and 'dataset_2'"
    assert len(fused[0]._parents) == 0, "Fused node should have no parents"


def test_no_multiple_parents_no_fusing():
    pipeline_one2many_fusing_possible = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1",
                name="first_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],
                outputs="dataset_3",
                name="child_node",
            ),
        ],
        tags=[NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("dummy")],
    )

    fused = fuse(pipeline_one2many_fusing_possible)

    assert len(fused) == len(
        pipeline_one2many_fusing_possible.nodes
    ), "No fusing has been performed, as child node can be fused to different parents."


def test_fusing_multiple_parents():
    pipeline_with_multiple_parents = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs=["dataset_1"],
                name="first_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_c",
                ],
                outputs="dataset_2",
                name="second_node",
            ),
            Node(
                func=dummy_fn,
                inputs=None,
                outputs="dataset_3",
                name="third_node",
            ),
            Node(
                func=dummy_fn,
                inputs=[
                    "dataset_1",
                    "dataset_2",
                ],
                outputs="dataset_4",
                name="child_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_3", "dataset_4"],
                outputs="dataset_5",
                name="grandchild_node",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_5"],
                outputs="dataset_6",
                name="grandgrandchild_node",
            ),
        ],
        tags=[NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("dummy")],
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


def dummy_func(*args) -> str:
    return "dummy"


@pytest.fixture()
def simple_node():
    return Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")


@pytest.fixture()
def fused_node():
    return FusedNode(depth=0)


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
    fused_node._nodes[0].tags = [NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("test")]

    fusable_node = Node(
        func=dummy_func,
        inputs=["dataset_c"],
        outputs="dataset_d",
        name="fusable_node",
        tags=[NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("test")],
    )

    assert fused_node.fuses_with(fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    non_fusable_node = Node(func=dummy_func, inputs=["dataset_x"], outputs="dataset_y", name="non_fusable_node")

    assert not fused_node.fuses_with(non_fusable_node)


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = [NodeTags.ARGO_FUSE_NODE.value]
    assert fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_not_is_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert not fused_node.is_fusable


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_fuse_group(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = [fuse_group_tag("test_group")]
    assert fused_node.fuse_group == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_nodes_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_func, name="second_node"))
    assert fused_node.nodes == "simple_node,second_node"


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_outputs_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node.add_node(Node(func=dummy_func, outputs="dataset_d"))
    assert fused_node.outputs == {"dataset_c", "dataset_d"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_tags_property(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = ["tag1", "tag2"]
    fused_node.add_node(Node(func=dummy_func, tags=["tag2", "tag3"]))
    assert fused_node.tags == {"tag1", "tag2", "tag3"}


# TODO(pascal.bro): Let's determine what the desired behaviour is
@pytest.mark.skip(reason="Desired behaviour not clear")
def test_name_property_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    fused_node._nodes[0].tags = [NodeTags.ARGO_FUSE_NODE.value, fuse_group_tag("test_group")]
    fused_node.add_node(Node(func=dummy_func, name="second_node"))
    assert fused_node.name == "test_group"


# TODO(pascal.bro): Let's determine what the desired behaviour is
def test_name_property_not_fusable(fused_node: FusedNode, simple_node: Node) -> None:
    fused_node.add_node(simple_node)
    assert fused_node.name == "simple_node"


def test_get_fuse_group() -> None:
    tags = [fuse_group_tag("test_group"), "other_tag"]
    assert FusedNode.get_fuse_group(tags) == "test_group"


def test_get_fuse_group_no_group() -> None:
    tags = ["other_tag"]
    assert FusedNode.get_fuse_group(tags) is None


def test_clean_dependencies() -> None:
    elements = ["dataset_a@pandas", "params:some_param", "dataset_b"]
    cleaned = FusedNode.clean_dependencies(elements)
    assert cleaned == ["dataset_a", "dataset_b"]


def test_get_k8s_node_affinity_tags_with_gpu():
    tags = ["some-tag", NodeTags.K8S_REQUIRE_GPU.value, "another-tag"]
    result = get_k8s_node_affinity_tags(tags)
    assert result == [NodeTags.K8S_REQUIRE_GPU.value]


def test_get_k8s_node_affinity_tags_without_gpu():
    tags = ["some-tag", "another-tag"]
    result = get_k8s_node_affinity_tags(tags)
    assert result == []


@pytest.fixture
def mock_fused_node():
    node = Mock()
    node.name = "test-node"
    node.nodes = ["node1", "node2"]
    node._parents = set()
    node.tags = ["tag1", "tag2"]
    return node


@pytest.fixture
def mock_fused_node_with_argo_tags():
    node = Mock()
    node.name = "test-node-argo"
    node.nodes = ["node1"]
    node._parents = set()
    node.tags = [f"{ARGO_NODE_PREFIX}memory-4Gi", f"{ARGO_NODE_PREFIX}cpu-2"]
    return node


@pytest.fixture
def mock_fused_node_with_parents(mock_fused_node):
    node = Mock()
    node.name = "child-node"
    node.nodes = ["node3"]
    node._parents = {mock_fused_node}
    node.tags = ["tag3"]
    return node


def test_empty_pipeline():
    result = get_pipeline_as_tasks([])
    assert result == []


def test_single_node_no_deps(mock_fused_node):
    result = get_pipeline_as_tasks([mock_fused_node])

    assert len(result) == 1
    assert result[0]["name"] == "test-node"
    assert result[0]["nodes"] == ["node1", "node2"]
    assert result[0]["deps"] == []
    assert result[0]["tags"] == ["tag1", "tag2"]


def test_node_with_argo_tags(mock_fused_node_with_argo_tags):
    result = get_pipeline_as_tasks([mock_fused_node_with_argo_tags])

    assert len(result) == 1
    assert result[0]["memory"] == "4Gi"
    assert result[0]["cpu"] == "2"


def test_multiple_nodes_with_deps(mock_fused_node, mock_fused_node_with_parents):
    result = get_pipeline_as_tasks([mock_fused_node, mock_fused_node_with_parents])

    assert len(result) == 2

    # Check parent node
    assert result[0]["name"] == "test-node"
    assert result[0]["deps"] == []

    # Check child node
    assert result[1]["name"] == "child-node"
    assert result[1]["deps"] == ["test-node"]


def test_k8s_affinity_tags(mock_fused_node):
    mock_affinity = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}}

    with patch("matrix.argo.get_k8s_node_affinity_tags", return_value=mock_affinity) as mock_get_tags:
        result = get_pipeline_as_tasks([mock_fused_node])

        mock_get_tags.assert_called_once_with(mock_fused_node.tags)
        assert result[0]["k8s_affinity_tags"] == mock_affinity


def assert_argo_config_structure(parsed_config: dict, expected_pipeline_names: list[str]) -> None:
    """Helper function to verify the structure of an Argo workflow config.

    Args:
        parsed_config: The parsed YAML configuration as a dictionary
        expected_pipeline_names: List of pipeline names that should be present in templates
    """
    assert isinstance(parsed_config, dict), "Parsed config should be a dictionary"

    # Verify spec
    spec = parsed_config["spec"]

    # Verify kedro template
    kedro_template = next(t for t in spec["templates"] if t["name"] == "kedro")
    assert kedro_template["backoff"]["duration"] == "1", "Kedro template should have correct backoff duration"
    assert kedro_template["backoff"]["factor"] == 2, "Kedro template should have correct backoff factor"
    assert kedro_template["backoff"]["maxDuration"] == "1m", "Kedro template should have correct max backoff duration"
    assert "nodeAntiAffinity" in kedro_template["affinity"], "Kedro template should have nodeAntiAffinity"
    assert kedro_template["metadata"]["labels"]["app"] == "kedro-argo", "Kedro template should have correct label"

    # Check if the pipeline is included in the templates
    templates = spec["templates"]
    pipeline_names = [template["name"] for template in templates]
    for name in expected_pipeline_names:
        assert name in pipeline_names, f"The '{name}' pipeline should be included in the templates"

    return spec


def test_generate_argo_config() -> None:
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    run_name = "test_run"
    image_tag = "test_tag"
    namespace = "test_namespace"
    username = "test_user"
    pipelines = {
        "test_pipeline": Pipeline(
            nodes=[Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")]
        ),
        "cloud_pipeline": Pipeline(
            nodes=[
                Node(
                    func=dummy_func,
                    inputs=["dataset_a_cloud", "dataset_b_cloud"],
                    outputs="dataset_c_cloud",
                    name="simple_node_cloud",
                )
            ]
        ),
    }

    argo_config = generate_argo_config(
        image=image_name,
        run_name=run_name,
        image_tag=image_tag,
        namespace=namespace,
        username=username,
        pipelines=pipelines,
        package_name="matrix",
    )

    assert argo_config is not None
    parsed_config = yaml.safe_load(argo_config)

    spec = assert_argo_config_structure(
        parsed_config=parsed_config, expected_pipeline_names=["test_pipeline", "cloud_pipeline"]
    )

    templates = spec["templates"]

    # Verify test_pipeline template
    test_template = next(t for t in templates if t["name"] == "test_pipeline")
    assert "dag" in test_template, "test_pipeline template should have a DAG"
    assert len(test_template["dag"]["tasks"]) == 1, "test_pipeline template should have one task"
    assert (
        test_template["dag"]["tasks"][0]["name"] == "simple-node"
    ), "test_pipeline template should have correct task name"
    assert (
        test_template["dag"]["tasks"][0]["template"] == "kedro"
    ), "test_pipeline template task should use kedro template"
    assert (
        "affinity" not in test_template["dag"]["tasks"][0]
    ), "test_pipeline template task should not have explicit affinity"

    # Verify cloud_pipeline template
    cloud_template = next(t for t in templates if t["name"] == "cloud_pipeline")
    assert "dag" in cloud_template, "cloud_pipeline template should have a DAG"
    assert len(cloud_template["dag"]["tasks"]) == 1, "cloud_pipeline template should have one task"
    assert (
        cloud_template["dag"]["tasks"][0]["name"] == "simple-node-cloud"
    ), "cloud_pipeline template should have correct task name"
    assert (
        cloud_template["dag"]["tasks"][0]["template"] == "kedro"
    ), "cloud_pipeline template task should use kedro template"
    assert (
        "affinity" not in cloud_template["dag"]["tasks"][0]
    ), "cloud_pipeline template task should not have explicit affinity"


def test_generate_argo_config_with_gpu_affinity() -> None:
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    run_name = "test_run"
    image_tag = "test_tag"
    namespace = "test_namespace"
    username = "test_user"
    pipelines = {
        "test_pipeline": Pipeline(
            nodes=[Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")]
        ),
        "cloud_pipeline": Pipeline(
            nodes=[
                Node(
                    func=dummy_func,
                    inputs=["dataset_a_cloud", "dataset_b_cloud"],
                    outputs="dataset_c_cloud",
                    name="simple_node_cloud",
                    tags=[NodeTags.K8S_REQUIRE_GPU.value],
                )
            ]
        ),
    }

    argo_config = generate_argo_config(
        image=image_name,
        run_name=run_name,
        image_tag=image_tag,
        namespace=namespace,
        username=username,
        pipelines=pipelines,
        package_name="matrix",
    )

    assert argo_config is not None
    parsed_config = yaml.safe_load(argo_config)

    spec = assert_argo_config_structure(
        parsed_config=parsed_config, expected_pipeline_names=["test_pipeline", "cloud_pipeline"]
    )

    templates = spec["templates"]

    # Verify test_pipeline template
    test_template = next(t for t in templates if t["name"] == "test_pipeline")
    assert "dag" in test_template, "test_pipeline template should have a DAG"
    assert len(test_template["dag"]["tasks"]) == 1, "test_pipeline template should have one task"
    assert (
        test_template["dag"]["tasks"][0]["name"] == "simple-node"
    ), "test_pipeline template should have correct task name"
    assert (
        test_template["dag"]["tasks"][0]["template"] == "kedro"
    ), "test_pipeline template task should use kedro template"
    assert (
        "affinity" not in test_template["dag"]["tasks"][0]
    ), "test_pipeline template task should not have explicit affinity"

    # Verify cloud_pipeline template
    cloud_template = next(t for t in templates if t["name"] == "cloud_pipeline")
    assert "dag" in cloud_template, "cloud_pipeline template should have a DAG"
    assert len(cloud_template["dag"]["tasks"]) == 1, "cloud_pipeline template should have one task"
    assert (
        cloud_template["dag"]["tasks"][0]["name"] == "simple-node-cloud"
    ), "cloud_pipeline template should have correct task name"
    assert (
        cloud_template["dag"]["tasks"][0]["template"] == "kedro"
    ), "cloud_pipeline template task should use kedro template"
    assert "affinity" in cloud_template["dag"]["tasks"][0], "cloud_pipeline template task should have explicit affinity"
