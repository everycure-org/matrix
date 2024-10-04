from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
import pytest

from matrix.argo import clean_name, fuse


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
def test_clean_name(input_name, expected):
    """Test clean_name function with various input cases."""
    assert clean_name(input_name) == expected
