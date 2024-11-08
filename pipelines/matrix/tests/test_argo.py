from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
import pytest
import yaml

from matrix.argo import clean_name, fuse, FusedNode, generate_argo_config


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
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
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
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
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
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
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
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test"]

    fusable_node = Node(
        func=dummy_func,
        inputs=["dataset_c"],
        outputs="dataset_d",
        name="fusable_node",
        tags=["argowf.fuse", "argowf.fuse-group.test"],
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
    fused_node._nodes[0].tags = ["argowf.fuse", "argowf.fuse-group.test_group"]
    fused_node.add_node(Node(func=dummy_func, name="second_node"))
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

    templates = spec["templates"]
    # Check if the pipeline is included in the templates
    pipeline_names = [template["name"] for template in templates]
    assert "test_pipeline" in pipeline_names, "The 'test_pipeline' pipeline should be included in the templates"
    assert "cloud_pipeline" in pipeline_names, "The 'cloud_pipeline' pipeline should be included in the templates"

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

    resources_no_gpu = kedro_template["container"]["resources"]
    # Verify no GPU resources when use_gpus=False
    assert resources_no_gpu["requests"]["nvidia.com/gpu"] == 0, "GPU request should be 0 when use_gpus=False"
    assert resources_no_gpu["limits"]["nvidia.com/gpu"] == 0, "GPU limit should be 0 when use_gpus=False"


def test_generate_argo_config_with_gpus() -> None:
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
        use_gpus=True,
    )

    assert argo_config is not None

    parsed_config = yaml.safe_load(argo_config)

    assert isinstance(parsed_config, dict), "Parsed config should be a dictionary"

    # Verify spec
    spec = parsed_config["spec"]

    # Verify kedro template
    kedro_template = next(t for t in spec["templates"] if t["name"] == "kedro")
    assert kedro_template["backoff"]["duration"] == "1", "Kedro template should have correct backoff duration"
    assert kedro_template["backoff"]["factor"] == 2, "Kedro template should have correct backoff factor"
    assert kedro_template["backoff"]["maxDuration"] == "1m", "Kedro template should have correct max backoff duration"
    assert "nodeAffinity" in kedro_template["affinity"], "Kedro template should have nodeAffinity"
    assert "nodeAntiAffinity" not in kedro_template["affinity"], "Kedro template should not have nodeAntiAffinity"
    assert (
        "requiredDuringSchedulingIgnoredDuringExecution" in kedro_template["affinity"]["nodeAffinity"]
    ), "Kedro template should have requiredDuringSchedulingIgnoredDuringExecution"
    selector = kedro_template["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"].get(
        "nodeSelectorTerms", []
    )
    assert len(selector) == 1, "Kedro template should have one node selector term"
    match_expression = selector[0]["matchExpressions"][0]
    assert match_expression["key"] == "gpu_node", "Kedro template should have correct GPU node selector key"
    assert match_expression["operator"] == "In", "Kedro template should have correct operator"
    assert match_expression["values"] == ["true"], "Kedro template should have correct GPU node selector value"

    templates = spec["templates"]
    # Check if the pipeline is included in the templates
    pipeline_names = [template["name"] for template in templates]
    assert "test_pipeline" in pipeline_names, "The 'test_pipeline' pipeline should be included in the templates"
    assert "cloud_pipeline" in pipeline_names, "The 'cloud_pipeline' pipeline should be included in the templates"

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

    resources = kedro_template["container"]["resources"]

    # Check requests
    assert resources["requests"]["memory"] == "64Gi", "Memory request should be 64Gi"
    assert resources["requests"]["cpu"] == 4, "CPU request should be 4"
    assert resources["requests"]["nvidia.com/gpu"] == 1, "GPU request should be 1"

    # Check limits
    assert resources["limits"]["memory"] == "64Gi", "Memory limit should be 64Gi"
    assert resources["limits"]["cpu"] == 16, "CPU limit should be 16"
    assert resources["limits"]["nvidia.com/gpu"] == 1, "GPU limit should be 1"
