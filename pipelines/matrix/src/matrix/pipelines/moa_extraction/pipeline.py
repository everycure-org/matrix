"""
MOA extraction pipeline.
"""
# TODO: Add mapping success report node

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from . import nodes


num_hops_lst = ["two", "three"]


def _preprocessing_pipeline() -> Pipeline:
    initial_nodes = pipeline(
        [
            node(
                func=nodes.add_tags,
                inputs={
                    "runner": "params:moa_extraction.neo4j_runner",
                    "drug_types": "params:moa_extraction.tagging_options.drug_types",
                    "disease_types": "params:moa_extraction.tagging_options.disease_types",
                    "batch_size": "params:moa_extraction.tagging_options.batch_size",
                    "verbose": "params:moa_extraction.tagging_options.verbose",
                },
                outputs=None,
                tags=["moa_extraction.preprocessing", "moa_extraction.tagging"],
                name="add_tags",
            ),
            node(
                func=nodes.get_one_hot_encodings,
                inputs={"runner": "params:moa_extraction.neo4j_runner"},
                outputs=["moa_extraction.feat.category_encoder", "moa_extraction.feat.relation_encoder"],
                name="get_one_hot_encodings",
                tags="moa_extraction.preprocessing",
            ),
        ]
    )
    preprocessing_strands_lst = []
    for num_hops in num_hops_lst:
        preprocessing_strands_lst.append(
            pipeline(
                [
                    node(
                        func=nodes.map_drug_mech_db,
                        inputs={
                            "runner": "params:moa_extraction.neo4j_runner",
                            "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                            "mapper": f"params:moa_extraction.path_mapping.mapper_{num_hops}_hop",
                            "synonymizer_endpoint": "params:moa_extraction.path_mapping.synonymizer_endpoint",
                        },
                        outputs=f"moa_extraction.int.{num_hops}_hop_indication_paths",
                        name=f"map_{num_hops}_hop",
                        tags="moa_extraction.preprocessing",
                    ),
                    node(
                        func=nodes.make_splits,
                        inputs={
                            "paths_data": f"moa_extraction.int.{num_hops}_hop_indication_paths",
                            "splitter": f"params:moa_extraction.splits.splitter_{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.prm.{num_hops}_hop_splits",
                        name=f"make_splits_{num_hops}_hop",
                        tags="moa_extraction.preprocessing",
                    ),
                ]
            )
        )
    return sum(
        [
            initial_nodes,
            *preprocessing_strands_lst,
        ]
    )


def _training_pipeline() -> Pipeline:
    training_strands_lst = []
    for num_hops in num_hops_lst:
        training_strands_lst.append(
            pipeline(
                [
                    node(
                        func=nodes.create_training_features,
                        inputs={
                            "splits_data": f"moa_extraction.prm.{num_hops}_hop_splits",
                        },
                        outputs=None,
                        name=f"create_features_{num_hops}_hop",
                        tags="moa_extraction.training",
                    ),
                    # node(
                    #     func=nodes.train_model,
                    #     inputs=["moa_extraction.prm.two_hop_splits", "moa_extraction.feat.category_encoder", "moa_extraction.feat.relation_encoder"],
                    #     outputs="moa_extraction.mdl.model",
                    #     name=f"train_model_{num_hops}_hop",
                    #     tags="moa_extraction.training",
                    # ),
                ]
            )
        )
    return sum(training_strands_lst)


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline() + _training_pipeline()
