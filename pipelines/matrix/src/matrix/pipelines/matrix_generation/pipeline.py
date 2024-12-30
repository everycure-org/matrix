from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ARGO_GPU_NODE_MEDIUM, ArgoNode

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""

    # Load model names
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]

    # Load cross-validation information
    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation")
    n_splits = cross_validation_settings.get("n_splits")
    folds_lst = list(range(n_splits)) + ["full"]

    # Initial nodes computing matrix pairs and flags
    pipelines = []

    # Add shared nodes
    pipelines.append(
        pipeline(
            [
                ArgoNode(
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
                ArgoNode(
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
    for fold in folds_lst:
        # For each fold, generate the pairs
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.generate_pairs,
                        inputs=[
                            "ingestion.raw.drug_list@pandas",
                            "ingestion.raw.disease_list@pandas",
                            "matrix_generation.feat.nodes_kg_ds",
                            f"modelling.model_input.fold_{fold}.splits",
                            "ingestion.raw.clinical_trials_data",
                        ],
                        outputs=f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                        name=f"generate_matrix_pairs_fold_{fold}",
                    )
                ]
            )
        )

        # For each model, create pipeline to generate pairs
        for model in model_names:
            pipelines.append(
                pipeline(
                    [
                        ArgoNode(
                            func=nodes.make_predictions_and_sort,
                            inputs=[
                                "matrix_generation.feat.nodes_kg_ds",
                                f"matrix_generation.prm.fold_{fold}.matrix_pairs",
                                f"modelling.{model}.fold_{fold}.model_input.transformers",
                                f"modelling.{model}.fold_{fold}.models.model",
                                f"params:modelling.{model}.model_options.model_tuning_args.features",
                                "params:evaluation.score_col_name",
                                "params:matrix_generation.matrix_generation_options.batch_by",
                            ],
                            outputs=f"matrix_generation.{model}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                            name=f"make_{model}_predictions_and_sort_fold_{fold}",
                            argo_config=ARGO_GPU_NODE_MEDIUM,
                        ),
                        ArgoNode(
                            func=nodes.generate_report,
                            inputs=[
                                f"matrix_generation.{model}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                                "params:matrix_generation.matrix_generation_options.n_reporting",
                                "ingestion.raw.drug_list@pandas",
                                "ingestion.raw.disease_list@pandas",
                                "params:evaluation.score_col_name",
                                "params:matrix_generation.matrix",
                                "params:matrix_generation.run",
                            ],
                            outputs=f"matrix_generation.{model}.fold_{fold}.reporting.matrix_report",
                            name=f"generate_{model}_report_fold_{fold}",
                        ),
                    ],
                    tags=model,
                )
            )

    return sum(pipelines)
