from kedro.pipeline import Pipeline
from matrix.kedro_extension import ArgoNode

# What would be an e2e test for ArgoPipeline?


# IN: Kedro pipeline, containing ArgoNodes which are supposed to be fused. Ideally something based on the tests that are already there.

# When Argo Pipeline is initialized (with the Kedro pipeline), it should automatically fuse the nodes which need fusing.

# Then, the only thing remaining is to render the Argo Template.

# OUT: Argo template, which can be rendered to an Argo workflow.

from matrix.argo_pipeline import ArgoPipeline


def test_argo_pipeline_without_fusing(parallel_pipelines):
    k8s_pipeline, _ = parallel_pipelines
    argo_pipeline = ArgoPipeline(k8s_pipeline)
    argo_kedro_command = argo_pipeline.kedro_command()
    argo_tasks = argo_pipeline.tasks

    assert argo_kedro_command is not None
    assert len(argo_tasks) == len(k8s_pipeline.nodes)


def dummy_fn(*args):
    return "dummy"


def test_no_nodes_fused_when_no_fuse_options():
    pipeline_with_no_fusing_options = Pipeline(
        nodes=[
            ArgoNode(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_c",
                name="first",
            ),
            ArgoNode(
                func=dummy_fn,
                inputs=["dataset_1", "dataset_2"],  # inputs are different than outputs of previous node
                outputs="dataset_3",
                name="second",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    argo_pipeline = ArgoPipeline(pipeline_with_no_fusing_options)
    argo_tasks = argo_pipeline.tasks

    assert all(isinstance(task, ArgoNode) for task in argo_tasks)
    assert len(argo_tasks) == len(
        pipeline_with_no_fusing_options.nodes
    ), "No nodes should be fused when no fuse options are provided"


def test_simple_fusing():
    pipeline_where_first_node_is_input_for_second = Pipeline(
        nodes=[
            ArgoNode(
                func=dummy_fn,
                inputs=["dataset_a", "dataset_b"],
                outputs="dataset_1@pandas",
            ),
            ArgoNode(
                func=dummy_fn,
                inputs=[
                    "dataset_1@spark",
                ],
                outputs="dataset_2",
            ),
        ],
        tags=["argowf.fuse", "argowf.fuse-group.dummy"],
    )

    argo_pipeline = ArgoPipeline(pipeline_where_first_node_is_input_for_second)
    argo_tasks = argo_pipeline.tasks
    assert all(isinstance(task, ArgoNode) for task in argo_tasks)

    assert len(argo_tasks) == 1, "Only one node should be fused"
    assert argo_tasks[0].name == "dummy", "Fused node should have name 'dummy'"
    assert argo_tasks[0].outputs == set(
        ["dataset_1", "dataset_2"]
    ), "Fused node should have outputs 'dataset_1' and 'dataset_2'"
    assert len(argo_tasks[0]._parents) == 0, "Fused node should have no parents"
