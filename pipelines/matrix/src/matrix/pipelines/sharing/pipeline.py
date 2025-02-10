import time

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from matrix import settings


def log_experiment(
    input_leaderboard: pd.DataFrame,
    evaluation_results_simple_classification: pd.DataFrame,
    evaluation_results_simple_classification_trials: pd.DataFrame,
    evaluation_results_disease_specific: pd.DataFrame,
    evaluation_results_disease_specific_trials: pd.DataFrame,
    evaluation_results_full_matrix: pd.DataFrame,
    evaluation_results_full_matrix_clinical: pd.DataFrame,
    log_params: dict,
) -> pd.DataFrame:
    time_str = time.strftime("%Y-%m-%d")
    new_exp = pd.DataFrame(
        {
            "RUN_NAME": log_params["RUN_NAME"],
            "EXPERIMENT_NAME": log_params["EXPERIMENT_NAME"],
            "DESCRIPTION": log_params["DESCRIPTION"],
            "DOCUMENT": log_params["DOCUMENT"],
            "TIER": log_params["TIER"],
            "RECALL_1000": evaluation_results_full_matrix["mean_recall-1000"],
            "RECALL_10000": evaluation_results_full_matrix["mean_recall-10000"],
            "RECALL_100000": evaluation_results_full_matrix["mean_recall-100000"],
            "MRR": evaluation_results_disease_specific["mean_mrr"],
            "F1": evaluation_results_simple_classification["mean_f1_score"],
            #'HIT_5': evaluation_results_disease_specific['mean_hit-5'],
            #'HIT_10': evaluation_results_disease_specific['mean_hit-10'],
            # 'HIT_100': evaluation_results_disease_specific['mean_hit-100'],
            # 'RECALL_1000_CLINICAL': evaluation_results_full_matrix_clinical['mean_recall-1000'],
            # 'RECALL_10000_CLINICAL': evaluation_results_full_matrix_clinical['mean_recall-10000'],
            # 'RECALL_100000_CLINICAL': evaluation_results_full_matrix_clinical['mean_recall-100000'],
            # 'MRR_CLINICAL': evaluation_results_disease_specific_trials['mean_mrr'],
            # 'F1_CLINICAL': evaluation_results_simple_classification_trials['mean_f1_score'],
            # 'HIT_5_CLINICAL': evaluation_results_disease_specific_trials['mean_hit-5'],
            # 'HIT_10_CLINICAL': evaluation_results_disease_specific_trials['mean_hit-10'],
            # 'HIT_100_CLINICAL': evaluation_results_disease_specific_trials['mean_hit-100'],
            "MODEL": log_params["MODEL"],
            "TIMESTAMP": time_str,
        }
    )
    return pd.concat([input_leaderboard, new_exp], ignore_index=True, axis=0)


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for sharing results with medical team."""
    return pipeline(
        [
            node(
                func=log_experiment,
                inputs={
                    "input_leaderboard": "sharing.reporting.input_leaderboard",
                    "evaluation_results_simple_classification": "evaluation.simple_classification.reporting.result_aggregated_reduced",
                    "evaluation_results_simple_classification_trials": "evaluation.simple_classification_trials.reporting.result_aggregated_reduced",
                    "evaluation_results_disease_specific": "evaluation.disease_specific.reporting.result_aggregated_reduced",
                    "evaluation_results_disease_specific_trials": "evaluation.disease_specific_trials.reporting.result_aggregated_reduced",
                    "evaluation_results_full_matrix": "evaluation.full_matrix.reporting.result_aggregated_reduced",
                    "evaluation_results_full_matrix_clinical": "evaluation.full_matrix_trials.reporting.result_aggregated_reduced",
                    "log_params": "params:sharing.reporting.experiment",
                },
                outputs=f"sharing.reporting.experiment",
                name="log_experiment_run",
            ),
        ]
    )
    # return sum([*pipe])
