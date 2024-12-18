from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.pipelines.modelling import nodes as modelling_nodes
from matrix.kedro4argo_node import argo_node

from . import nodes


def _create_evaluation_pipeline(model: str, evaluation: str, fold: str) -> Pipeline:
    return pipeline(
        [
            argo_node(
                func=nodes.generate_test_dataset,
                inputs=[
                    f"matrix_generation.{model}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                ],
                outputs=f"evaluation.{model}.fold_{fold}.{evaluation}.model_output.pairs",
                name=f"create_{model}_{evaluation}_evaluation_pairs_fold_{fold}",
            ),
            argo_node(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.fold_{fold}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result",
                name=f"create_{model}_{evaluation}_evaluation_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.{evaluation}.fold_{fold}"],
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline."""
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_names = [model["model_name"] for model in models]
    n_splits = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_splits")
    folds_lst = list(range(n_splits))
    evaluation_names = [
        evaluation["evaluation_name"] for evaluation in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")
    ]

    pipelines = []
    for model in model_names:
        for fold in folds_lst:
            pipelines.append(
                pipeline(
                    [
                        argo_node(
                            func=nodes.perform_matrix_checks,
                            inputs=[
                                f"matrix_generation.{model}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                                f"modelling.model_input.fold_{fold}.splits",
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
            return ["params:modelling.aggregation_functions"] + [
                f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result" for fold in range(n_splits)
            ]

        for evaluation in evaluation_names:
            pipelines.append(
                pipeline(
                    [
                        argo_node(
                            func=modelling_nodes.aggregate_metrics,
                            inputs=_give_aggregation_node_input(model),
                            outputs=f"evaluation.{model}.{evaluation}.reporting.result_aggregated",
                            name=f"aggregate_{model}_{evaluation}_evaluation_results",
                            tags=[model, evaluation],
                        ),
                        # Reduce the aggregate results for simpler readout in MLFlow (e.g. only report mean)
                        argo_node(
                            func=nodes.reduce_aggregated_results,
                            inputs=[
                                f"evaluation.{model}.{evaluation}.reporting.result_aggregated",
                                "params:evaluation.reported_aggregations",
                            ],
                            outputs=f"evaluation.{model}.{evaluation}.reporting.result_aggregated_reduced",
                            name=f"reduce_aggregated_{model}_{evaluation}_evaluation_results",
                            tags=[model, evaluation],
                        ),
                    ]
                )
            )

    # Generate dictionaries of inputs for the consolidation node
    collect_fold_specific_reports = {
        model + "." + evaluation + ".fold_" + str(fold): f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result"
        for model in model_names
        for evaluation in evaluation_names
        for fold in folds_lst
    }
    collect_aggregated_reports = {
        model + "." + evaluation + ".aggregated": f"evaluation.{model}.{evaluation}.reporting.result_aggregated"
        for model in model_names
        for evaluation in evaluation_names
    }

    pipelines.append(
        pipeline(
            [
                argo_node(
                    func=nodes.consolidate_evaluation_reports,
                    inputs={**collect_fold_specific_reports, **collect_aggregated_reports},
                    outputs="evaluation.reporting.master_report",
                    name="consolidate_evaluation_reports",
                )
            ]
        )
    )

    return sum(pipelines)
