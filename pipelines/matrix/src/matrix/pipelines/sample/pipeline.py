from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs=["integration.prm.original.filtered_edges"],
                outputs="integration.prm.filtered_edges",
                name="sample_filtered_edges",
            )
            # TODO: Add other nodes for sampling
        ]
    )
