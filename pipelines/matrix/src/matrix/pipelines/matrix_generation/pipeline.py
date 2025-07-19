from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import (
    ARGO_NODE_MEDIUM_MATRIX_GENERATION,
    ArgoNode,
)
from matrix.pipelines.modelling.utils import partial_fold

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create matrix generation pipeline."""

    model_name = settings.DYNAMIC_PIPELINES_MAPPING().get("modelling")["model_name"]

    # Load cross-validation information
    cross_validation_settings = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation")
    n_cross_val_folds = cross_validation_settings.get("n_cross_val_folds")

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
                        "integration.int.drug_list.nodes.norm@spark",
                        "integration.int.disease_list.nodes.norm@spark",
                    ],
                    outputs="matrix_generation.feat.nodes@spark",
                    name="enrich_matrix_embeddings",
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
                    ArgoNode(
                        func=partial_fold(nodes.generate_pairs, fold, arg_name="known_pairs"),
                        inputs={
                            "known_pairs": "modelling.model_input.splits@pandas",
                            "drugs": "integration.int.drug_list.nodes.norm@pandas",
                            "diseases": "integration.int.disease_list.nodes.norm@pandas",
                            "graph": "matrix_generation.feat.nodes@kg",
                            "clinical_trials": "integration.int.ec_clinical_trails.edges.norm@pandas",
                            "off_label": "integration.int.off_label.edges.norm@pandas",
                        },
                        outputs=f"matrix_generation.prm.fold_{fold}.matrix_pairs@pandas",
                        name=f"generate_matrix_pairs_fold_{fold}",
                    )
                ]
            )
        )

        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.make_predictions_and_sort,
                        inputs=[
                            "matrix_generation.feat.nodes@spark",
                            f"matrix_generation.prm.fold_{fold}.matrix_pairs@spark",
                            f"modelling.fold_{fold}.model_input.transformers",
                            f"modelling.fold_{fold}.models.model",
                            # TODO: can we get features from transformers directly?
                            f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                            "params:matrix_generation.treat_score_col_name",
                            "params:matrix_generation.not_treat_score_col_name",
                            "params:matrix_generation.unknown_score_col_name",
                        ],
                        outputs=f"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@spark",
                        name=f"make_predictions_and_sort_fold_{fold}",
                        argo_config=ARGO_NODE_MEDIUM_MATRIX_GENERATION,
                    ),
                ],
            )
        )

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.generate_reports,
                    inputs=[
                        f"matrix_generation.fold_{n_cross_val_folds}.model_output.sorted_matrix_predictions@pandas",
                        "params:matrix_generation.reporting_nodes.plots",
                    ],
                    outputs="matrix_generation.reporting.plots",
                    name="generate_reporting_plots",
                ),
                ArgoNode(
                    func=nodes.generate_reports,
                    inputs={
                        "sorted_matrix_df": f"matrix_generation.fold_{n_cross_val_folds}.model_output.sorted_matrix_predictions@spark",
                        "strategies": "params:matrix_generation.reporting_nodes.tables",
                        "drugs_df": "integration.int.drug_list.nodes.norm@spark",
                        "diseases_df": "integration.int.disease_list.nodes.norm@spark",
                    },
                    outputs="matrix_generation.reporting.tables",
                    name="generate_reporting_tables",
                ),
            ]
        )
    )

    return sum(pipelines)
