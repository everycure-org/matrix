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
    argo_tasks = argo_pipeline.tasks()

    assert argo_kedro_command is not None
    assert len(argo_tasks) == len(k8s_pipeline.nodes)
