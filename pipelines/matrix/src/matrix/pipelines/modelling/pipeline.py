"""Modelling pipeline."""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from matrix import settings

from . import nodes


def _create_model_shard_pipeline(model: str, shard: int) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "modelling.model_input.splits",
                    f"params:modelling.{model}.model_options.generator",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.enriched_splits",
                name=f"enrich_{model}_{shard}_splits",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.enriched_splits",
                    f"modelling.{model}.model_input.transformers",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.transformed_splits",
                name=f"transform_{model}_{shard}_data",
            ),
            node(
                func=nodes.tune_parameters,
                inputs={
                    "data": f"modelling.{model}.{shard}.model_input.transformed_splits",
                    "unpack": f"params:modelling.{model}.model_options.model_tuning_args",
                },
                outputs=[
                    f"modelling.{model}.{shard}.models.model_params",
                    f"modelling.{model}.{shard}.reporting.tuning_convergence_plot",
                ],
                name=f"tune_model_{model}_{shard}_parameters",
            ),
            node(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.transformed_splits",
                    f"modelling.{model}.{shard}.models.model_params",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{model}.{shard}.models.model",
                name=f"train_{model}_{shard}_model",
            ),
        ]
    )


def _create_model_pipeline(model: str, num_shards: int) -> Pipeline:
    return sum(
        [
            pipeline(
                [
                    node(
                        func=nodes.fit_transformers,
                        inputs=[
                            "modelling.model_input.splits",
                            f"params:modelling.{model}.model_options.transformers",
                        ],
                        outputs=f"modelling.{model}.model_input.transformers",
                        name=f"fit_{model}_transformers",
                        tags=model,
                    )
                ]
            ),
            *[
                pipeline(
                    _create_model_shard_pipeline(model=model, shard=shard),
                    tags=model,
                )
                for shard in range(num_shards)
            ],
            pipeline(
                [
                    node(
                        func=nodes.create_model,
                        inputs=[
                            f"modelling.{model}.{shard}.models.model"
                            for shard in range(num_shards)
                        ],
                        outputs=f"modelling.{model}.models.model",
                        name=f"create_{model}_model",
                        tags=model,
                    ),
                    node(
                        func=nodes.apply_transformers,
                        inputs=[
                            f"modelling.model_input.splits",
                            f"modelling.{model}.model_input.transformers",
                        ],
                        outputs=f"modelling.{model}.model_input.transformed_splits",
                        name=f"transform_{model}_data",
                    ),
                    node(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.{model}.model_input.transformed_splits",
                            "model": f"modelling.{model}.models.model",
                            "features": f"params:modelling.{model}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.model_output.predictions",
                        name=f"get_{model}_model_predictions",
                    ),
                    node(
                        func=nodes.check_model_performance,
                        inputs={
                            "data": f"modelling.{model}.model_output.predictions",
                            "metrics": f"params:modelling.{model}.model_options.metrics",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.reporting.metrics",
                        name=f"check_{model}_model_performance",
                    ),
                ]
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline."""
    create_model_input = pipeline(
        [
            node(
                func=nodes.prefilter_nodes,
                inputs=[
                    "embeddings.feat.nodes",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.model_input.drugs_diseases_nodes@spark",
                name="prefilter_nodes",
            ),
            node(
                func=nodes.create_feat_nodes,
                inputs=[
                    "modelling.model_input.drugs_diseases_nodes@spark",
                    "integration.model_input.ground_truth",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.feat.rtx_kg2",
                name="create_feat_nodes",
            ),
            node(
                func=nodes.make_splits,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "integration.model_input.ground_truth",
                    "params:modelling.splitter",
                ],
                outputs="modelling.model_input.splits",
                name="create_splits",
            ),
        ]
    )

    pipes = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        pipes.append(
            pipeline(
                _create_model_pipeline(
                    model=model["model_name"], num_shards=model["num_shards"]
                ),
                tags=model["model_name"],
            )
        )

    return sum([create_model_input, *pipes])
