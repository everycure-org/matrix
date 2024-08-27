from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline

from matrix.argo import fuse


def dummy_fn(*args):
    return "dummy"


def test_no_fusing():
    # Given a pipeline with no fusing options
    pipeline = Pipeline(
        nodes=[
            Node(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="first",
            ),
            Node(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],
                outputs="dataset_3",
                name="second",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    # When applying fusing to the pipeline
    fused = fuse(pipeline)

    # Then no nodes fused
    assert len(fused) == len(pipeline.nodes)


def test_simple_fusing():
    # Given a pipeline where first nodes provides
    # all inputs for second node
    pipeline = Pipeline(
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

    # When applying fusing to the pipeline
    fused = fuse(pipeline)

    # Then nodes fused correctly into a singe node,
    # fuse group is used to name the fused entity, outputs
    # of fused node is union of outputs of the input nodes
    # and the fused node has no parents.
    assert len(fused) == 1
    assert fused[0].name == "dummy"
    assert fused[0].outputs == set(["dataset_1", "dataset_2"])
    assert len(fused[0]._parents) == 0


def test_no_multiple_parents_no_fusing():
    # Given a pipeline where child node can be fused to multiple
    # parents
    pipeline = Pipeline(
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

    # When applying fusing to the pipeline
    fused = fuse(pipeline)

    # No fusing has been performed, as child node can be fused
    # to different parents.
    assert len(fused) == len(pipeline.nodes)


def test_fusing_multiple_parents():
    pipeline = Pipeline(
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
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    # When applying fusing to the pipeline
    fused = fuse(pipeline)

    # Fusing of child and grandchild node, ensure correct naming
    # and recording of parent relationships and dataset outputs.
    assert len(fused) == 4
    assert fused[3].name == "dummy"
    assert fused[3].outputs == set(["dataset_4", "dataset_5"])
    assert set([parent.name for parent in fused[3]._parents]) == set(
        ["first_node", "second_node", "third_node"]
    )
