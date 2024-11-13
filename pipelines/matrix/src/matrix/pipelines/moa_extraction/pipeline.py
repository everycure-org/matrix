"""
MOA extraction pipeline.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

import matrix.pipelines.embeddings.nodes as embeddings_nodes

from . import nodes
from matrix import settings

moa_extraction_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("moa_extraction")
num_hops_lst = [model["num_hops"] for model in moa_extraction_settings]


def _preprocessing_pipeline() -> Pipeline:
    initial_nodes = pipeline(
        [
            node(
                func=nodes.get_one_hot_encodings,
                inputs=["integration.prm.filtered_nodes", "integration.prm.filtered_edges"],
                outputs=["moa_extraction.feat.category_encoder", "moa_extraction.feat.relation_encoder"],
                name="get_one_hot_encodings",
                tags="moa_extraction.preprocessing",
            ),
        ],
    )
    preprocessing_strands_lst = []
    for num_hops in num_hops_lst:
        preprocessing_strands_lst.append(
            pipeline(
                [
                    node(
                        func=embeddings_nodes.ingest_nodes,
                        inputs=["integration.prm.filtered_nodes"],
                        outputs=f"moa_extraction.input_nodes.{num_hops}_hop",
                        name=f"moa_extraction_ingest_neo4j_input_nodes_{num_hops}_hop",
                        tags=["moa_extraction.create_neo4j_db"],
                    ),
                    node(
                        func=embeddings_nodes.ingest_edges,
                        inputs=[f"moa_extraction.input_nodes.{num_hops}_hop", "integration.prm.filtered_edges"],
                        outputs=f"moa_extraction.input_edges.{num_hops}_hop",
                        name=f"ingest_neo4j_input_edges_{num_hops}_hop",
                        tags=["moa_extraction.create_neo4j_db"],
                    ),
                    node(
                        func=nodes.add_tags,
                        inputs={
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "drug_types": "params:moa_extraction.tagging_options.drug_types",
                            "disease_types": "params:moa_extraction.tagging_options.disease_types",
                            "batch_size": "params:moa_extraction.tagging_options.batch_size",
                            "verbose": "params:moa_extraction.tagging_options.verbose",
                            "edges": f"moa_extraction.input_edges.{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.reporting.add_tags_{num_hops}_hop",
                        tags=["moa_extraction.tagging"],
                        name=f"add_tags_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.map_drug_mech_db,
                        inputs={
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                            "mapper": f"params:moa_extraction.path_mapping.mapper_{num_hops}_hop",
                            "drugmechdb_entities": "moa_extraction.raw.drugmechdb_entities",
                            "add_tags_dummy": f"moa_extraction.reporting.add_tags_{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.int.{num_hops}_hop_indication_paths",
                        name=f"map_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.report_mapping_success,
                        inputs={
                            "drugmechdb_entities": "moa_extraction.raw.drugmechdb_entities",
                            "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                            "mapped_paths": f"moa_extraction.int.{num_hops}_hop_indication_paths",
                        },
                        outputs=f"moa_extraction.reporting.{num_hops}_hop_mapping_success",
                        name=f"report_mapping_success_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.make_splits,
                        inputs={
                            "paths_data": f"moa_extraction.int.{num_hops}_hop_indication_paths",
                            "splitter": f"params:moa_extraction.splits.splitter_{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.prm.{num_hops}_hop_splits",
                        name=f"make_splits_{num_hops}_hop",
                    ),
                ],
                tags=[
                    "argowf.fuse",
                    f"argowf.fuse-group.moa_extraction_{num_hops}_hop",
                    "argowf.template-neo4j",
                    "moa_extraction.preprocessing",
                ],
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
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                        name=f"generate_negative_paths_{num_hops}_hop",
                        tags="moa_extraction.negative_sampling",
                    ),
                    node(
                        func=nodes.train_model_split,
                        inputs={
                            "tuner": f"params:moa_extraction.training.{num_hops}_hop.tuner",
                            "paths": f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.models.{num_hops}_hop_model_split",
                        name=f"train_{num_hops}_hop_model_split",
                    ),
                    node(
                        func=nodes.train_model,
                        inputs={
                            "tuner": f"params:moa_extraction.training.{num_hops}_hop.tuner",
                            "paths": f"moa_extraction.feat.{num_hops}_hop_enriched_paths",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.models.{num_hops}_hop_model",
                        name=f"train_{num_hops}_hop_model",
                    ),
                ],
                tags=[
                    "argowf.fuse",
                    f"argowf.fuse-group.moa_extraction_{num_hops}_hop",
                    "argowf.template-neo4j",
                    "moa_extraction.training",
                ],
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
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "positive_paths": f"moa_extraction.prm.{num_hops}_hop_splits",
                            "path_generator": f"params:moa_extraction.evaluation.{num_hops}_hop.path_generator",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                        },
                        outputs=f"moa_extraction.model_output.{num_hops}_hop_evaluation_predictions",
                        name=f"evaluation.make_{num_hops}_hop_predictions",
                    ),
                    node(
                        func=nodes.compute_evaluation_metrics,
                        inputs={
                            "positive_paths": f"moa_extraction.prm.{num_hops}_hop_splits",
                            "predictions": f"moa_extraction.model_output.{num_hops}_hop_evaluation_predictions",
                            "k_lst": f"params:moa_extraction.evaluation.{num_hops}_hop.k_lst",
                        },
                        outputs=f"moa_extraction.reporting.{num_hops}_hop_metrics",
                        name=f"compute_{num_hops}_hop_metrics",
                    ),
                ],
                tags=[
                    "argowf.fuse",
                    f"argowf.fuse-group.moa_extraction_{num_hops}_hop",
                    "argowf.template-neo4j",
                    "moa_extraction.evaluation",
                ],
            )
        )
    return sum(evaluation_strands_lst)


def _predictions_pipeline() -> Pipeline:
    predictions_strands_lst = []
    for num_hops in num_hops_lst:
        predictions_strands_lst.append(
            pipeline(
                [
                    node(
                        func=nodes.make_output_predictions,
                        inputs={
                            "model": f"moa_extraction.models.{num_hops}_hop_model",
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "pairs": "moa_extraction.raw.pairs_for_moa_prediction",
                            "path_generator": f"params:moa_extraction.predictions.{num_hops}_hop.path_generator",
                            "path_embedding_strategy": "params:moa_extraction.path_embeddings.strategy",
                            "category_encoder": "moa_extraction.feat.category_encoder",
                            "relation_encoder": "moa_extraction.feat.relation_encoder",
                            "drug_col_name": "params:moa_extraction.predictions.drug_col_name",
                            "disease_col_name": "params:moa_extraction.predictions.disease_col_name",
                            "num_pairs_limit": "params:moa_extraction.predictions.num_pairs_limit",
                        },
                        outputs=f"moa_extraction.model_output.{num_hops}_hop_output_predictions",
                        name=f"predictions.make_{num_hops}_hop_output_predictions",
                    ),
                    node(
                        func=nodes.generate_predictions_reports,
                        inputs={
                            "predictions": f"moa_extraction.model_output.{num_hops}_hop_output_predictions",
                            "include_edge_directions": "params:moa_extraction.predictions.include_edge_directions",
                            "num_paths_per_pair_limit": "params:moa_extraction.predictions.num_paths_per_pair_limit",
                        },
                        outputs={
                            "excel_reports": f"moa_extraction.reporting.{num_hops}_hop_predictions_report",
                            "pair_info_dfs": f"moa_extraction.model_output.{num_hops}_hop_pair_info_sql",
                            "moa_predictions_dfs": f"moa_extraction.model_output.{num_hops}_hop_predictions_sql",
                        },
                        name=f"generate_{num_hops}_hop_predictions_report",
                    ),
                ],
                tags=[
                    "argowf.fuse",
                    f"argowf.fuse-group.moa_extraction_{num_hops}_hop",
                    "argowf.template-neo4j",
                    "moa_extraction.predictions",
                ],
            )
        )
    return sum(predictions_strands_lst)


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline() + _training_pipeline() + _evaluation_pipeline() + _predictions_pipeline()
