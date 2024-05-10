from kedro.pipeline import Pipeline, node, pipeline


from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.create_prm_pairs,
                inputs=[
                    "raw.experiments.tp",
                    "raw.experiments.tn",
                ],
                outputs="prm.known_pairs",
                name="create_int_known_pairs"
            ),
            node(
                func=nodes.create_feat_nodes,
                inputs=[
                    "raw.rtx_kg2.nodes",
                    "params:drug_types",
                    "params:disease_types",
                    "raw.fda_drugs"
                ],
                outputs="feat.rtx_kg2",
                name="create_feat_nodes"
            ),
            node(
                func=nodes.make_splits,
                inputs=[
                    "prm.known_pairs", 
                    "params:splitter"
                ],
                outputs="model_input.splits",
                name="create_splits"
            ),
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "feat.rtx_kg2", 
                    "prm.known_pairs",
                    "params:generator"
                ],
                outputs="model_input.unkown_pairs",
                name="create_model_input_nodes"
            ),
        ]
    )