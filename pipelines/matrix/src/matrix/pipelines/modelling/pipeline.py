from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings

from . import nodes


def _create_model_shard_pipeline(model: str, shard: int, fold: int) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "modelling.model_input.drugs_diseases_nodes@pandas",
                    f"modelling.model_input.splits_fold_{fold}",
                    f"params:modelling.{model}.model_options.generator",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.enriched_splits_fold_{fold}",
                name=f"enrich_{model}_{shard}_splits_fold_{fold}",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.enriched_splits_fold_{fold}",
                    f"modelling.{model}.model_input.transformers_fold_{fold}",
                ],
                outputs=f"modelling.{model}.{shard}.model_input.transformed_splits_fold_{fold}",
                name=f"transform_{model}_{shard}_data_fold_{fold}",
            ),
            node(
                func=nodes.tune_parameters,
                inputs={
                    "data": f"modelling.{model}.{shard}.model_input.transformed_splits_fold_{fold}",
                    "unpack": f"params:modelling.{model}.model_options.model_tuning_args",
                },
                outputs=[
                    f"modelling.{model}.{shard}.models.model_params_fold_{fold}",
                    f"modelling.{model}.{shard}.reporting.tuning_convergence_plot_fold_{fold}",
                ],
                name=f"tune_model_{model}_{shard}_parameters_fold_{fold}",
            ),
            node(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{model}.{shard}.model_input.transformed_splits_fold_{fold}",
                    f"modelling.{model}.{shard}.models.model_params_fold_{fold}",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{model}.{shard}.models.model_fold_{fold}",
                name=f"train_{model}_{shard}_model_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.shard-{shard}.fold-{fold}"],
    )


def _create_model_pipeline(model: str, num_shards: int, fold: int) -> Pipeline:
    return sum(
        [
            pipeline(
                [
                    node(
                        func=nodes.fit_transformers,
                        inputs=[
                            f"modelling.model_input.splits_fold_{fold}",
                            f"params:modelling.{model}.model_options.transformers",
                        ],
                        outputs=f"modelling.{model}.model_input.transformers_fold_{fold}",
                        name=f"fit_{model}_transformers_fold_{fold}",
                        tags=model,
                    )
                ]
            ),
            *[
                pipeline(
                    _create_model_shard_pipeline(model=model, shard=shard, fold=fold),
                    tags=model,
                )
                for shard in range(num_shards)
            ],
            pipeline(
                [
                    node(
                        func=nodes.create_model,
                        inputs=[f"modelling.{model}.{shard}.models.model_fold_{fold}" for shard in range(num_shards)],
                        outputs=f"modelling.{model}.models.model_fold_{fold}",
                        name=f"create_{model}_model_fold_{fold}",
                        tags=model,
                    ),
                    node(
                        func=nodes.apply_transformers,
                        inputs=[
                            f"modelling.model_input.splits_fold_{fold}",
                            f"modelling.{model}.model_input.transformers_fold_{fold}",
                        ],
                        outputs=f"modelling.{model}.model_input.transformed_splits_fold_{fold}",
                        name=f"transform_{model}_data_fold_{fold}",
                    ),
                    node(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.{model}.model_input.transformed_splits_fold_{fold}",
                            "model": f"modelling.{model}.models.model_fold_{fold}",
                            "features": f"params:modelling.{model}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.model_output.predictions_fold_{fold}",
                        name=f"get_{model}_model_predictions_fold_{fold}",
                    ),
                ],
                tags=["argowf.fuse", f"argowf.fuse-group.{model}.fold-{fold}"],
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline."""
    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation")
    n_splits = cross_validation_settings.get("n_splits")

    folds_lst = [fold for fold in range(n_splits)] + ["full"]
    models_lst = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names_lst = [model["model_name"] for model in models_lst]

    create_model_input = pipeline(
        [
            # Construct ground_truth
            node(
                func=nodes.create_int_pairs,
                inputs=[
                    "embeddings.feat.nodes",
                    "modelling.raw.ground_truth.positives@spark",
                    "modelling.raw.ground_truth.negatives@spark",
                ],
                outputs="modelling.int.known_pairs@spark",
                name="create_int_known_pairs",
            ),
            node(
                func=nodes.prefilter_nodes,
                inputs=[
                    "embeddings.feat.nodes",
                    "modelling.raw.ground_truth.positives@spark",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.model_input.drugs_diseases_nodes@spark",
                name="prefilter_nodes",
            ),
            node(
                func=nodes.make_splits,
                inputs=[
                    "modelling.int.known_pairs@pandas",
                    "params:modelling.splitter",
                ],
                outputs=[f"modelling.model_input.splits_fold_{fold}" for fold in folds_lst],
                name="create_splits",
            ),
        ],
        tags=[model for model in model_names_lst],
    )

    # Compute metrics on all folds but not model trained on full data
    check_performance = pipeline(
        [
            node(
                func=nodes.check_model_performance,
                inputs={
                    "data": f"modelling.{model}.model_output.predictions_fold_{fold}",
                    "metrics": f"params:modelling.{model}.model_options.metrics",
                    "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                },
                outputs=f"modelling.{model}.reporting.metrics_fold_{fold}",
                name=f"check_{model}_model_performance_fold_{fold}",
                tags=[f"{model}", "argowf.fuse", f"argowf.fuse-group.{model}.fold-{fold}"],
            )
            for fold in range(n_splits)
            for model in model_names_lst
        ]
    )

    # Aggregation step
    def _give_aggregation_node_input(model):
        """Prepare aggregation node inputs, including reports for all folds"""
        return ["params:modelling.aggregation_functions"] + [
            f"modelling.{model}.reporting.metrics_fold_{fold}" for fold in range(n_splits)
        ]

    aggregate_metrics = pipeline(
        [
            node(
                func=nodes.aggregate_metrics,
                inputs=_give_aggregation_node_input(model),
                outputs=f"modelling.{model}.reporting.metrics_aggregated",
                name=f"aggregate_{model}_model_performance_checks",
                tags=[f"{model}"],
            )
            for model in model_names_lst
        ]
    )

    pipelines = []
    for fold in folds_lst:
        for model in models_lst:
            pipelines.append(
                pipeline(
                    _create_model_pipeline(model=model["model_name"], num_shards=model["num_shards"], fold=fold),
                    tags=[model["model_name"], "not-shared"],
                )
            )
    return sum([create_model_input, *pipelines, check_performance, aggregate_metrics])
