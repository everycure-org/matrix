"""
MOA extraction pipeline.
"""
# TODO: Inject prefix into add_tags and AllPathsWithTagRules
# TODO: Consider moving neo query helpers top their own file
# TODO: Make negative path sampling strategies PathGenerator Strategies
# TODO: Simplify the schema by rename source and target
# TODO: Add mapping success report node
# TODO: Add training curve plot to MLFlow report
# TODO: Onehot to sklearn
# TODO: Replace neo4j runner by Kedro node

# TODO: reconsider class structure for path generators and negative samplers

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
                        func=nodes.generate_negative_paths,
                        inputs={
                            "paths": f"moa_extraction.prm.{num_hops}_hop_splits",
                            "negative_sampler_list": f"params:moa_extraction.training.{num_hops}_hop.negative_samplers",
                            "runner": "params:moa_extraction.neo4j_runner",
                        },
                        outputs=f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                        name=f"generate_negative_paths_{num_hops}_hop",
                        tags=["moa_extraction.training", "moa_extraction.negative_sampling"],
                    ),
                    node(
                        func=nodes.train_model_split,
                        inputs={
                            "model": f"params:moa_extraction.training.{num_hops}_hop.model",
                            "paths": f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.models.{num_hops}_hop_model_split",
                        name=f"train_{num_hops}_hop_model_split",
                        tags="moa_extraction.training",
                    ),
                    node(
                        func=nodes.train_model,
                        inputs={
                            "model": f"params:moa_extraction.training.{num_hops}_hop.model",
                            "paths": f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.models.{num_hops}_hop_model",
                        name=f"train_{num_hops}_hop_model",
                        tags="moa_extraction.training",
                    ),
                ]
            )
        )
    return sum(training_strands_lst)


def _evaluation_pipeline() -> Pipeline:
    evaluation_strands_lst = []
    for num_hops in num_hops_lst:
        evaluation_strands_lst.append(
            pipeline(
                [
                    node(
                        func=nodes.make_evaluation_predictions,
                        inputs={
                            "model": f"moa_extraction.models.{num_hops}_hop_model",
                            "runner": "params:moa_extraction.neo4j_runner",
                            "positive_paths": f"moa_extraction.prm.{num_hops}_hop_splits",
                            "path_generator": f"params:moa_extraction.evaluation.{num_hops}_hop.path_generators",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.evaluation.{num_hops}_hop_predictions",
                        name=f"moa_extraction.evaluation.make_{num_hops}_hop_predictions",
                        tags=["moa_extraction.evaluation"],
                    ),
                    node(
                        func=nodes.compute_evaluation_metrics,
                        inputs={
                            "positive_paths": f"moa_extraction.prm.{num_hops}_hop_splits",
                            "predictions": f"moa_extraction.evaluation.{num_hops}_hop_predictions",
                            "k_lst": f"params:moa_extraction.evaluation.{num_hops}_hop.k_lst",
                        },
                        outputs=f"moa_extraction.evaluation.{num_hops}_hop_metrics",
                        name=f"moa_extraction.evaluation.compute_{num_hops}_hop_metrics",
                        tags=["moa_extraction.evaluation"],
                    ),
                ]
            )
        )
    return sum(evaluation_strands_lst)


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline() + _training_pipeline() + _evaluation_pipeline()
