"""Module with functions for model evaluation."""
from typing import Dict, List
import pandas as pd
import numpy as np

import bisect

from sklearn.metrics import roc_auc_score, average_precision_score

from matrix.datasets.drp_model import DRPmodel

## TO DO: acc, F1
## remove types from docstring


def get_training_metrics(
    drp_model: DRPmodel,
    data: pd.DataFrame,
    metrics: List[callable],
    target_col_name: str,
) -> Dict:
    """Function to evaluate model performance on training data.

    TO DO: modify to include AUROC classification score

    Args:
        drp_model (DRPmodel): Model giving a probability score.
        data (pd.DataFrame): Data to evaluate.
        metrics (List[callable]): List of callable metrics.
        target_col_name (str): Target column name.

    Returns:
        Dictionary containing report
    """
    # Restrict to training data only
    is_train = data["split"].eq("TRAIN")
    train_data = data[is_train]

    # Compute probability score
    treat_scores = drp_model.give_treat_scores(train_data, skip_vectorise=True)[
        "treat score"
    ]
    # True label
    y_true = train_data[target_col_name]

    report = {}

    for metric in metrics:
        # Execute metric
        report[f"{metric.__name__}"] = metric(y_true == 1, treat_scores > 0.5).item()

    return report


def get_classification_metrics(
    drp_model: DRPmodel,
    data: pd.DataFrame,
    metrics: List[callable],
    target_col_name: str,
) -> Dict:
    """Function to evaluate model performance on known test data.

    TO DO: modify to include AUROC classification score

    Args:
        drp_model (DRPmodel): Model giving a probability score.
        data (pd.DataFrame): Data to evaluate.
        metrics (List[callable]): List of callable metrics.
        target_col_name (str): Target column name.

    Returns:
        Dictionary containing report
    """
    # Restrict to test data only
    is_test = data["split"].eq("TEST")
    test_data = data[is_test]

    # Compute probability score
    treat_scores = drp_model.give_treat_scores(test_data, skip_vectorise=True)[
        "treat score"
    ]
    # True label
    y_true = test_data[target_col_name]

    report = {}

    for metric in metrics:
        # Execute metric
        report[f"{metric.__name__}"] = metric(y_true == 1, treat_scores > 0.5).item()

    return report


def perform_disease_centric_evaluation(
    drp_model: DRPmodel,
    known_data: pd.DataFrame,
    k_lst: List[int],
    target_col_name: str,
) -> Dict:
    """Function to perform disease centric evaluation.

    This version will use drugs with known positives
    TO DO: docstring using the time-split analysis notebook
    """
    # Extracting known positive test data and training data
    is_test = known_data["split"].eq("TEST")
    is_pos = known_data[target_col_name].eq(1)
    kp_data_test = known_data[is_test & is_pos]
    train_data = known_data[~is_test]

    # List of drugs over which to rank
    all_drugs = pd.Series(kp_data_test["source"].unique())
    # Diseases appearing the known positive test set
    test_diseases = pd.Series(kp_data_test["target"].unique())

    # Loop over test diseases
    mrr_total = 0
    hitk_total_lst = np.zeros(len(k_lst))
    df_all_exists = False
    for disease in list(test_diseases):
        # Extract relevant train and test datapoints
        all_pos_test = kp_data_test[kp_data_test["target"] == disease]
        all_train = train_data[train_data["target"] == disease]

        # Construct negatives
        check_cond_pos_test = lambda drug: drug not in list(all_pos_test["source"])
        check_cond_train = lambda drug: drug not in list(all_train["source"])
        check_conds = lambda drug: check_cond_pos_test(drug) and check_cond_train(drug)
        negative_drugs = all_drugs[all_drugs.map(check_conds)]
        negative_pairs = pd.DataFrame({"source": negative_drugs, "target": disease})
        negative_pairs[target_col_name] = 0

        # Compute probability scores
        all_pos_test = drp_model.give_treat_scores(all_pos_test, skip_vectorise=True)
        negative_pairs = drp_model.give_treat_scores(negative_pairs)

        # Concatenate to DataFrame with all probability scores
        if df_all_exists:
            df_all = pd.concat(
                (df_all, all_pos_test, negative_pairs), ignore_index=True
            )
        else:
            df_all = pd.concat((all_pos_test, negative_pairs), ignore_index=True)
            df_all_exists = True

        # Compute rank for all positives
        negative_pairs = negative_pairs.sort_values("treat score", ascending=True)
        for _, prob in all_pos_test["treat score"].items():
            rank = (
                len(negative_pairs)
                - bisect.bisect_left(list(negative_pairs["treat score"]), prob)
                + 1
            )
            # Add to total
            mrr_total += 1 / rank
            for idx, k in enumerate(k_lst):
                if rank <= k:
                    hitk_total_lst[idx] += 1

    mrr = mrr_total / len(kp_data_test)
    hitk_lst = list(hitk_total_lst / len(kp_data_test))

    # Computing AUROC and AP
    y_true = df_all[target_col_name]
    y_score = df_all["treat score"]
    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    report = {
        "AUROC": auroc,
        "AP": ap,
        "MRR": mrr,
        "Hit@k": hitk_lst,
    }  # TO DO: fix format
    return report
