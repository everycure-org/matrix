from kedro.pipeline import Pipeline, node, pipeline


def process(nodes_df, edges_df):
    breakpoint()


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process,
                inputs=["integration.int.rtx-kg2.nodes.norm", "integration.int.rtx-kg2.edges.norm"],
                outputs="",
                name="create",
            )
        ]
    )
