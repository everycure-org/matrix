from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings

from . import nodes


def _create_matrix_generation_pipeline(model: str, fold: int) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.make_predictions_and_sort,
                inputs=[
                    "matrix_generation.feat.nodes_kg_ds",
                    "matrix_generation.prm.matrix_pairs",
                    f"modelling.{model}.model_input.transformers_fold_{fold}",
                    f"modelling.{model}.models.model_fold_{fold}",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    "params:evaluation.score_col_name",
                    "params:matrix_generation.matrix_generation_options.batch_by",
                ],
                outputs=f"matrix_generation.{model}.model_output.sorted_matrix_predictions_fold_{fold}@pandas",
                name=f"make_{model}_predictions_and_sort_fold_{fold}",
            ),
            node(
                func=nodes.generate_report,
                inputs=[
                    f"matrix_generation.{model}.model_output.sorted_matrix_predictions_fold_{fold}@pandas",
                    "params:matrix_generation.matrix_generation_options.n_reporting",
                    "ingestion.raw.drug_list@pandas",
                    "ingestion.raw.disease_list@pandas",
                    "params:evaluation.score_col_name",
                    "params:matrix_generation.matrix",
                    "params:matrix_generation.run",
                ],
                outputs=f"matrix_generation.{model}.reporting.matrix_report_fold_{fold}",
                name=f"generate_{model}_report_fold_{fold}",
            ),
        ],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""
    initial_nodes = pipeline(
        [
            node(
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
            node(
                func=nodes.spark_to_pd,
                inputs=[
                    "matrix_generation.feat.nodes@spark",
                ],
                outputs="matrix_generation.feat.nodes_kg_ds",
                name="transform_parquet_library",
            ),
            node(
                func=nodes.generate_pairs,
                inputs=[
                    "ingestion.raw.drug_list@pandas",
                    "ingestion.raw.disease_list@pandas",
                    "matrix_generation.feat.nodes_kg_ds",
                    "modelling.model_input.splits",
                    "ingestion.raw.clinical_trials_data",
                ],
                outputs="matrix_generation.prm.matrix_pairs",
                name="generate_matrix_pairs",
            ),
        ]
    )
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]

    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation")
    n_splits = cross_validation_settings.get("n_splits")
    folds_lst = [fold for fold in range(n_splits)] + ["full"]

    pipelines = [initial_nodes]
    for fold in folds_lst:
        for model in model_names:
            pipelines.append(
                pipeline(
                    _create_matrix_generation_pipeline(model, fold),
                    tags=model,
                )
            )

    return sum(pipelines)
