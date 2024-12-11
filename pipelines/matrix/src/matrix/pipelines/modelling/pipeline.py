from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ARGO_GPU_NODE_MEDIUM, argo_node

from . import nodes


def _create_model_shard_pipeline(model: str, shard: int) -> Pipeline:
    return pipeline(
        [
            argo_node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "modelling.model_input.drugs_diseases_nodes@pandas",
                    "modelling.model_input.splits",
                    f"params:modelling.{model}.model_options.generator",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.enriched_splits",
                name=f"enrich_{model}_{shard}_splits",
            ),
            argo_node(
                func=nodes.apply_transformers,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.enriched_splits",
                    f"modelling.{model}.model_input.transformers",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.transformed_splits",
                name=f"transform_{model}_{shard}_data",
            ),
            argo_node(
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
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
            argo_node(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.transformed_splits",
                    f"modelling.{model}.{shard}.models.model_params",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{model}.{shard}.models.model",
                name=f"train_{model}_{shard}_model",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.shard-{shard}"],
    )


def _create_model_pipeline(model: str, num_shards: int) -> Pipeline:
    return sum(
        [
            pipeline(
                [
                    argo_node(
                        func=nodes.fit_transformers,
                        inputs=[
                            "modelling.model_input.splits",
                            f"params:modelling.{model}.model_options.transformers",
                        ],
                        outputs=f"modelling.{model}.model_input.transformers",
                        name=f"fit_{model}_transformers",
                        tags=model,
                        argo_config=ARGO_GPU_NODE_MEDIUM,
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
                    argo_node(
                        func=nodes.create_model,
                        inputs=[f"modelling.{model}.{shard}.models.model" for shard in range(num_shards)],
                        outputs=f"modelling.{model}.models.model",
                        name=f"create_{model}_model",
                        tags=model,
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                    argo_node(
                        func=nodes.apply_transformers,
                        inputs=[
                            "modelling.model_input.splits",
                            f"modelling.{model}.model_input.transformers",
                        ],
                        outputs=f"modelling.{model}.model_input.transformed_splits",
                        name=f"transform_{model}_data",
                    ),
                    argo_node(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.{model}.model_input.transformed_splits",
                            "model": f"modelling.{model}.models.model",
                            "features": f"params:modelling.{model}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.model_output.predictions",
                        name=f"get_{model}_model_predictions",
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                    argo_node(
                        func=nodes.check_model_performance,
                        inputs={
                            "data": f"modelling.{model}.model_output.predictions",
                            "metrics": f"params:modelling.{model}.model_options.metrics",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.reporting.metrics",
                        name=f"check_{model}_model_performance",
                    ),
                ],
                tags=["argowf.fuse", f"argowf.fuse-group.{model}"],
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline."""
    create_model_input = pipeline(
        [
            # Construct ground_truth
            argo_node(
                func=nodes.filter_valid_pairs,
                inputs=[
                    "integration.prm.filtered_nodes_no_gt",
                    "ingestion.raw.ground_truth.combined@spark",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs={"pairs": "modelling.raw.known_pairs@spark", "metrics": "modelling.reporting.gt_present"},
                name="filter_valid_pairs",
            ),
            argo_node(
                func=nodes.attach_embeddings,
                inputs=[
                    "modelling.raw.known_pairs@spark",
                    "embeddings.feat.nodes",
                ],
                outputs="modelling.int.known_pairs@spark",
                name="create_int_known_pairs",
            ),
            argo_node(
                func=nodes.prefilter_nodes,
                inputs=[
                    "integration.prm.filtered_nodes",
                    "embeddings.feat.nodes",
                    "modelling.raw.known_pairs@spark",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.model_input.drugs_diseases_nodes@spark",
                name="prefilter_nodes",
            ),
            argo_node(
                func=nodes.make_splits,
                inputs=[
                    "modelling.int.known_pairs@pandas",
                    "params:modelling.splitter",
                ],
                outputs="modelling.model_input.splits",
                name="create_splits",
            ),
        ],
        tags=[model["model_name"] for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")],
    )

    pipelines = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        pipelines.append(
            pipeline(
                _create_model_pipeline(model=model["model_name"], num_shards=model["num_shards"]),
                tags=[model["model_name"], "not-shared"],
            )
        )

    return sum([create_model_input, *pipelines])
