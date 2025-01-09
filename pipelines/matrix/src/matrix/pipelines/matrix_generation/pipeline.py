from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.pipelines.modelling.utils import partial_fold
from matrix.kedro4argo_node import ARGO_GPU_NODE_MEDIUM, argo_node


from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""

    # Load model names
    model_name = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")["model_name"]

    # Load cross-validation information
    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation")
    n_cross_val_folds = cross_validation_settings.get("n_cross_val_folds")

    # Initial nodes computing matrix pairs and flags
    pipelines = []

    # Add shared nodes
    pipelines.append(
        pipeline(
            [
                argo_node(
                    func=nodes.enrich_embeddings,
                    inputs=[
                        "embeddings.feat.nodes",
                        "ingestion.raw.drug_list@spark",
                        "ingestion.raw.disease_list@spark",
                    ],
                    outputs="matrix_generation.feat.nodes@spark",
                    name="enrich_matrix_embeddings",
                ),
                # Hacky fix to save parquet file via pandas rather than spark
                # related to https://github.com/everycure-org/matrix/issues/71
                argo_node(
                    func=nodes.spark_to_pd,
                    inputs=[
                        "matrix_generation.feat.nodes@spark",
                    ],
                    outputs="matrix_generation.feat.nodes_kg_ds",
                    name="transform_parquet_library",
                ),
            ]
        )
    )

    # Nodes generating scores for each fold and model
    for fold in range(n_cross_val_folds + 1):  # NOTE: final fold is full training data
        # For each fold, generate the pairs
        pipelines.append(
            pipeline(
                [
                    argo_node(
                        func=partial_fold(nodes.generate_pairs, fold, arg_name="known_pairs"),
                        inputs={
                            "known_pairs": "modelling.model_input.splits",
                            "drugs": "ingestion.raw.drug_list@pandas",
                            "diseases": "ingestion.raw.disease_list@pandas",
                            "graph": "matrix_generation.feat.nodes_kg_ds",
                            "clinical_trials": "ingestion.raw.clinical_trials_data",
                        },
                        outputs=f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                        name=f"generate_matrix_pairs_fold_{fold}",
                    )
                ]
            )
        )

        pipelines.append(
            pipeline(
                [
                    argo_node(
                        func=nodes.make_predictions_and_sort,
                        inputs=[
                            "matrix_generation.feat.nodes_kg_ds",
                            f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                            f"modelling.{model_name}.fold_{fold}.model_input.transformers",
                            f"modelling.{model_name}.fold_{fold}.models.model",
                            f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                            "params:matrix_generation.treat_score_col_name",
                            "params:matrix_generation.not_treat_score_col_name",
                            "params:matrix_generation.unknown_score_col_name",
                            "params:matrix_generation.matrix_generation_options.batch_by",
                        ],
                        outputs=f"matrix_generation.{model_name}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                        name=f"make_{model_name}_predictions_and_sort_fold_{fold}",
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                    argo_node(
                        func=nodes.generate_report,
                        inputs=[
                            f"matrix_generation.{model_name}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                            "params:matrix_generation.matrix_generation_options.n_reporting",
                            "ingestion.raw.drug_list@pandas",
                            "ingestion.raw.disease_list@pandas",
                            "params:matrix_generation.treat_score_col_name",
                            "params:matrix_generation.matrix",
                            "params:matrix_generation.run",
                        ],
                        outputs=f"matrix_generation.{model_name}.fold_{fold}.reporting.matrix_report",
                        name=f"generate_{model_name}_report_fold_{fold}",
                    ),
                ],
                tags=model_name,
            )
        )

    return sum(pipelines)
