from kedro.pipeline import Pipeline, node, pipeline
from matrix import settings
from matrix.pipelines.modelling import nodes as modelling_nodes

from . import nodes


def _create_evaluation_pipeline(model: str, evaluation: str, fold: str) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.generate_test_dataset,
                inputs=[
                    f"matrix_generation.{model}.model_output.sorted_matrix_predictions_fold_{fold}@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                ],
                outputs=f"evaluation.{model}.{evaluation}.model_output.pairs_fold_{fold}",
                name=f"create_{model}_{evaluation}_evaluation_pairs_fold_{fold}",
            ),
            node(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.{evaluation}.model_output.pairs_fold_{fold}",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.{evaluation}.reporting.result_fold_{fold}",
                name=f"create_{model}_{evaluation}_evaluation_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.{evaluation}"],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline."""
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    n_splits = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_splits")
    folds_lst = [fold for fold in range(n_splits)]
    evaluation_names = [
        evaluation["evaluation_name"] for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")
    ]

    pipelines = []
    for model in model_names:
        for fold in folds_lst:
            pipelines.append(
                pipeline(
                    [
                        node(
                            func=nodes.perform_matrix_checks,
                            inputs=[
                                f"matrix_generation.{model}.model_output.sorted_matrix_predictions_fold_{fold}@pandas",
                                f"modelling.model_input.splits_fold_{fold}",
                                "params:evaluation.score_col_name",
                            ],
                            outputs=None,
                            name=f"perform_{model}_matrix_checks_fold_{fold}",
                            tags=["matrix_checks", model],
                        )
                    ]
                )
            )
            for evaluation in evaluation_names:
                pipelines.append(
                    pipeline(
                        _create_evaluation_pipeline(model, evaluation, fold),
                        tags=[model, evaluation],
                    )
                )

        # Aggregation step
        def _give_aggregation_node_input(model):
            """Prepare aggregation node inputs, including reports for all folds"""
            return ["params:modelling.agg_func"] + [
                f"evaluation.{model}.{evaluation}.reporting.result_fold_{fold}" for fold in range(n_splits)
            ]

        for evaluation in evaluation_names:
            pipelines.append(
                pipeline(
                    [
                        node(
                            func=modelling_nodes.aggregate_metrics,
                            inputs=_give_aggregation_node_input(model),
                            outputs=f"evaluation.{model}.{evaluation}.reporting.result_aggregated",
                            name=f"aggregate_{model}_{evaluation}_evaluation_results",
                            tags=[model, evaluation],
                        )
                    ]
                )
            )

    return sum(pipelines)
