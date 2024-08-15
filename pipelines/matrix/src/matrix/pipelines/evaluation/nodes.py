"""Module with nodes for evaluation."""
import json
from typing import Any, List, Dict, Union, Tuple

import sklearn.metrics as skl
from sklearn.impute._base import _BaseImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import _extract_elements_in_list

from matrix import settings
from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import DrugDiseasePairGenerator

from matrix.pipelines.preprocessing.nodes import resolve_curie
from matrix.pipelines.modelling.nodes import apply_transformers
from matrix.pipelines.evaluation.evaluation import Evaluation
from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.evaluation.utils import perform_disease_centric_evaluation


@has_schema(
    schema={
        "source": "object",
        "target": "object",
        "y": "int",
    },
    allow_subset=True,
)
@inject_object()
def generate_test_dataset(
    graph: KnowledgeGraph,
    known_pairs: pd.DataFrame,
    generator: DrugDiseasePairGenerator,
) -> pd.DataFrame:
    """Function to generate test dataset.

    Function leverages the given strategy to construct
    pairs dataset.

    Args:
        graph: KnowledgeGraph instance
        known_pairs: Labelled ground truth drug-disease pairs dataset.
        generator: Generator strategy
    Returns:
        Pairs dataframe
    """
    return generator.generate(graph, known_pairs)


def make_test_predictions(
    graph: KnowledgeGraph,
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    model: ModelWrapper,
    features: List[str],
    score_col_name: str,
) -> pd.DataFrame:
    """Generated probability scores for drug-disease dataset.

    TODO: Parallelise for large datasets

    Args:
        graph: Knowledge graph.
        data: Data to predict scores for.
        transformers: Dictionary of trained transformers.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        score_col_name: Probability score column name.

    Returns:
        Pairs dataset with additional column containing the probability scores.
    """
    # Collect embedding vectors
    data["source_embedding"] = data.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    data["target_embedding"] = data.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )

    # Apply transformers to data
    transformed = apply_transformers(data, transformers)
    features = _extract_elements_in_list(transformed.columns, features, raise_exc=True)

    # Generate model probability scores
    transformed[score_col_name] = model.predict_proba(transformed[features].values)[
        :, 1
    ]
    return transformed


@inject_object()
def evaluate_test_predictions(data: pd.DataFrame, evaluation: Evaluation) -> Any:
    """Function to apply evaluation.

    Args:
        data: predictions to evaluate on
        evaluation: metric to evaluate.

    Returns:
        Evaluation report
    """
    return evaluation.evaluate(data)


def consolidate_evaluation_reports(*reports) -> dict:
    """Function to consolidate evaluation reports into master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """
    reports_lst = [*reports]
    master_report = dict()
    for idx_1, model in enumerate(settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")):
        master_report[model["model_name"]] = dict()
        for idx_2, evaluation in enumerate(
            settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")
        ):
            master_report[model["model_name"]][
                evaluation["evaluation_name"]
            ] = reports_lst[idx_1 + idx_2]
    return json.loads(json.dumps(master_report, default=float))


def clean_clinical_trial_data(
    clinical_trial_data: pd.DataFrame, endpoint: str
) -> pd.DataFrame:
    """Function to clean clinical trials dataset for use in time-split evaluation metrics.

       Clinical trial data should be a EXCEL format containg 8 columns:
       Clinical Trial #, Reason for Rejection, Mapped Drug(s), Mapped Disease, Significantly Better?, Non-Significantly Better?, Non-Significantly Worse?, Significantly Worse?

       1. We filter out rows with the following conditions:
            - Missing Mapped Drug(s) name
            - Missing Mapped Disease name
            - with reason for rejection
            - missing values in either significantly better, non-significantly better, non-significantly worse, or significantly worse columns

       2. Map the disease_id and drug_id to the corresponding curie ids and filter out rows without mapping ids.
            - add column drug_kg_id
            - add column disease_kg_id

       3. Standardize the column names to the following:
            - Mapped Drug(s) name -> drug_name
            - Mapped Disease -> disease_name
            - Significantly Better? -> significantly_better
            - Non-Significantly Better? -> non_significantly_better
            - Non-Significantly Worse? -> non_significantly_worse
            - Significantly Worse? -> significantly_worse

       4. Only keep the following columns:
            - drug_name
            - disease_name
            - drug_kg_id
            - disease_kg_id
            - significantly_better
            - non_significantly_better
            - non_significantly_worse
            - significantly_worse

    Args:
        clinical_trial_data: Clinical trial data provided by medical team for evaluation.
        endpoint: endpoint of the synonymizer.

    Returns:
        Cleaned clinical trial data.
    """
    # rename all columns to make consistent
    clinical_trial_data.columns = [
        "clinical_trial_id",
        "reason_for_rejection",
        "drug_name",
        "disease_name",
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # remove rows with reason for rejection
    clinical_trial_data = clinical_trial_data[
        clinical_trial_data["reason_for_rejection"].map(lambda x: type(x) != str)
    ].reset_index(drop=True)

    # remove rows with missing drug or disease name
    row_has_missing = (
        clinical_trial_data["drug_name"].isna()
        | clinical_trial_data["disease_name"].isna()
    )
    clinical_trial_data = clinical_trial_data[~row_has_missing].reset_index(drop=True)

    # remove rows with missing values in significantly better, non-significantly better, non-significantly worse, or significantly worse columns
    row_has_missing = (
        clinical_trial_data["significantly_better"].isna()
        | clinical_trial_data["non_significantly_better"].isna()
        | clinical_trial_data["non_significantly_worse"].isna()
        | clinical_trial_data["significantly_worse"].isna()
    )
    clinical_trial_data = clinical_trial_data[~row_has_missing].reset_index(drop=True)

    # map drug_name and disease_name to drug_id and disease_id
    drug_names = clinical_trial_data["drug_name"].tolist()
    drug_kg_id = map(lambda x: resolve_curie(x, endpoint), drug_names)
    clinical_trial_data["drug_kg_id"] = drug_kg_id
    disease_names = clinical_trial_data["disease_name"].tolist()
    disease_kg_id = map(lambda x: resolve_curie(x, endpoint), disease_names)
    clinical_trial_data["disease_kg_id"] = disease_kg_id

    # remove rows with missing drug_id or disease_id
    row_has_missing = (
        clinical_trial_data["drug_kg_id"].isna()
        | clinical_trial_data["disease_kg_id"].isna()
    )
    clinical_trial_data = clinical_trial_data[~row_has_missing].reset_index(drop=True)

    # drop columns: clinical_trial_id, reason_for_rejection
    clinical_trial_data = clinical_trial_data.drop(
        columns=["clinical_trial_id", "reason_for_rejection"]
    )

    return clinical_trial_data[
        [
            "drug_name",
            "disease_name",
            "drug_kg_id",
            "disease_kg_id",
            "significantly_better",
            "non_significantly_better",
            "non_significantly_worse",
            "significantly_worse",
        ]
    ]


def _predict_scores(
    graph: KnowledgeGraph,
    model: ModelWrapper,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    cleaned_clinical_trial_data: pd.DataFrame,
) -> pd.DataFrame:
    """Function to generate predicted scores for clinical trial data.

    Args:
        graph: Knowledge graph.
        model: Model making the predictions.
        transformers: Dictionary of trained transformers.
        clinical_trial_data: Cleanned clinical trial data.

    Returns:
        Cleanned clinical trial data with additional column containing the probability scores.
    """
    # Collect embedding vectors
    cleaned_clinical_trial_data["source_embedding"] = cleaned_clinical_trial_data.apply(
        lambda row: graph._embeddings[row.drug_kg_id], axis=1
    )
    cleaned_clinical_trial_data["target_embedding"] = cleaned_clinical_trial_data.apply(
        lambda row: graph._embeddings[row.disease_kg_id], axis=1
    )

    # Apply transformers to data
    cleaned_clinical_trial_data = apply_transformers(
        cleaned_clinical_trial_data, transformers
    )

    # Generate model probability scores
    concatenated_rows = np.array(
        [
            np.concatenate(row)
            for row in cleaned_clinical_trial_data[
                ["source_embedding", "target_embedding"]
            ].values
        ]
    )
    predicted_scores = model.predict_proba(concatenated_rows)
    num_categories = len(model._estimators[0].classes_)
    if num_categories == 2:
        # we assume first is negative class and second is positive class
        predicted_scores_df = pd.DataFrame(
            predicted_scores, columns=["not treat score", "treat score"]
        )
        cleaned_clinical_trial_data = pd.concat(
            [
                cleaned_clinical_trial_data.drop(
                    columns=["source_embedding", "target_embedding"]
                ),
                predicted_scores_df,
            ],
            axis=1,
        )
    else:
        # we assume first, second and third are negative, positive and unknown class respectively
        predicted_scores_df = pd.DataFrame(
            predicted_scores,
            columns=["not treat score", "treat score", "unknown score"],
        )
        cleaned_clinical_trial_data = pd.concat(
            [
                cleaned_clinical_trial_data.drop(
                    columns=["source_embedding", "target_embedding"]
                ),
                predicted_scores_df,
            ],
            axis=1,
        )

    return cleaned_clinical_trial_data


@inject_object()
def generate_time_split_validation_barplot(
    graph: KnowledgeGraph,
    model: ModelWrapper,
    model_name: str,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    cleaned_clinical_trial_data: pd.DataFrame,
) -> plt.Figure:
    """Function to generate evaluation barplot.

    Args:
        graph: Knowledge graph.
        model: Model making the predictions.
        model_name: Name of the model.
        transformers: Dictionary of trained transformers.
        clinical_trial_data: Clinical trial data.

    Returns:
        fig: Generated matplotlib figure.
    """
    # Category and score names
    cat_name_lst = [
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
    ]
    score_type_lst = ["not treat score", "treat score", "unknown score"]

    ## Generate predicted scores for clinical trial data
    cleaned_clinical_trial_data = _predict_scores(
        graph, model, transformers, cleaned_clinical_trial_data
    )

    # Boolean series for splitting scores into outcome categories
    is_trials_category_lst = [
        cleaned_clinical_trial_data[cat_name] == 1 for cat_name in cat_name_lst
    ]

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    for n, score_type in enumerate(score_type_lst):
        probs_current_type = cleaned_clinical_trial_data[score_type]
        score_series_lst = [
            probs_current_type[is_trials_category]
            for is_trials_category in is_trials_category_lst
        ]
        quantiles = [
            list(score_series.quantile([0.25, 0.5, 0.75]))
            for score_series in score_series_lst
        ]
        heights = [quantile[1] for quantile in quantiles]
        lower_err = [quantile[1] - quantile[0] for quantile in quantiles]
        higher_err = [quantile[2] - quantile[1] for quantile in quantiles]
        yerr = [lower_err, higher_err]
        ax[n].bar([1, 2, 3], heights, yerr=yerr, zorder=-10, capsize=10)
        for m, score_series in enumerate(score_series_lst):
            ax[n].scatter(
                (m + 1) * np.ones(len(score_series)),
                score_series,
                color="orange",
                s=5,
                alpha=0.5,
            )
        ax[n].set_xticks([1, 2, 3])
        ax[n].set_xticklabels(cat_name_lst, rotation=45, ha="right", size=8)
        ax[n].set_ylabel(score_type_lst[n], size=12)
    fig.suptitle(model_name)

    return fig


@inject_object()
def _generate_time_split_validation_classification_auroc(
    graph: KnowledgeGraph,
    model: ModelWrapper,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    cleaned_clinical_trial_data: pd.DataFrame,
    categ_pos: str,
    categ_neg: str,
) -> float:
    """Computes an AUROC score which measures the probability that a drug-disease pair with a positive
    clinical trial outcome will have a higher "treat" score compared to a drug-disease pair with a
    worse clinical trial outcome. For example, "Significantly Better?" vs. "Non-Significantly Worse?".

    The score is computed by labelling the DD-pairs with better clinical trials outcomes as
    positives (y=1) and those with worse clinical trials outcomes as negatives (y=0).

    Args:
        graph: Knowledge graph.
        model: Model making the predictions.
        transformers: Dictionary of trained transformers.
        cleaned_clinical_trial_data: Clinical trial data.
        categ_pos: Column name for better clinical trials outcomes.
        categ_neg: Column name for worse clinical trials outcomes.

    Returns:
        AUROC score.
    """
    ## Computing treat scores for DD pairs with better and worse clinical trials outcomes respectively
    # Extract drug-disease pairs with better clinical trials outcomes
    cleaned_clinical_trial_data_pos = cleaned_clinical_trial_data[
        cleaned_clinical_trial_data[categ_pos] == 1
    ]

    # Generate predicted scores for clinical trial data with better clinical trials outcomes
    cleaned_clinical_trial_data_pos = _predict_scores(
        graph, model, transformers, cleaned_clinical_trial_data_pos
    )
    treat_probs_pos = cleaned_clinical_trial_data_pos["treat score"].to_list()

    # Extract drug-disease pairs with worse clinical trials outcomes
    cleaned_clinical_trial_data_neg = cleaned_clinical_trial_data[
        cleaned_clinical_trial_data[categ_neg] == 1
    ]

    # Generate predicted scores for clinical trial data with worse clinical trials outcomes
    cleaned_clinical_trial_data_neg = _predict_scores(
        graph, model, transformers, cleaned_clinical_trial_data_neg
    )
    treat_probs_neg = cleaned_clinical_trial_data_neg["treat score"].to_list()

    # Compute AUROC
    y_score = np.concatenate([treat_probs_pos, treat_probs_neg])
    y_true = np.concatenate(
        [np.ones_like(treat_probs_pos), np.zeros_like(treat_probs_neg)]
    )

    return skl.metrics.roc_auc_score(y_true, y_score)


@inject_object()
def generate_time_split_validation_classification_auroc(
    graph: KnowledgeGraph,
    model: ModelWrapper,
    model_name: str,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    cleaned_clinical_trial_data: pd.DataFrame,
) -> Tuple[List[float], List[str], str]:
    """Function to generate AUROC scores for time-split evaluation.

    Args:
        graph: Knowledge graph.
        model: Model making the predictions.
        model_name: Name of the model.
        transformers: Dictionary of trained transformers.
        clinical_trial_data: Clinical trial data.

    Returns:
        A tuple containing the AUROC scores for each pair of categories, a list of metric names, and the model name.
    """
    # Category names
    cat_name_lst = [
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
    ]

    # Compute AUROC scores for each pair of categories
    metrics_lst = []
    metric_name_lst = []
    for i, categ_1 in enumerate(cat_name_lst):
        for j, categ_2 in enumerate(cat_name_lst):
            if i < j:
                metrics_lst.append(
                    _generate_time_split_validation_classification_auroc(
                        graph,
                        model,
                        transformers,
                        cleaned_clinical_trial_data,
                        categ_1,
                        categ_2,
                    )
                )
                metric_name_lst.append(
                    'AUROC ("' + categ_1 + '" vs. "' + categ_2 + '")'
                )

    return (metrics_lst, metric_name_lst, model_name)


@inject_object()
def generate_time_split_validation_all_metrics(
    graph: KnowledgeGraph,
    model: ModelWrapper,
    model_name: str,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    cleaned_clinical_trial_data: pd.DataFrame,
    train_data: pd.DataFrame,
    k_lst: List[int],
    categ_pos: str,
    interested_drug_nodes: pd.DataFrame = None,
) -> Tuple[List[float], List[str], str]:
    """Returns a list of time-split clinical trials evaluation metrics of the form [auroc, ap, mrr] + hitk_lst,
    where:
        - (auroc) AUROC, approved drugs x test disease. Measures the probability that a drug-disease pair with a positive clinical
        trial outcome ranks higher than a synthetic negative.  The score is computed by labelling the DD-pairs
        with positive clinical trials outcomes as positives (y=1) and and all other DD-pairs in the
        approved drugs x test diseases matrix as negatives (y=0). Note that test diseases refers to
        the set of diseases appearing in the positively labelled DD-pairs and that we take care to remove
        all training data out of the matrix.
        - (ap) AP, approved drugs x test disease. AP score for drug-diseases labelled as in the above AUROC score.
        - (mrr) Disease-specific MRR score. Measures the mean reciprocal rank of DD-pairs with positive clinical
        trials outcomes when we fix the disease and rank over all approved drug nodes.
        - (hitk_list) Disease-specific Hit@k scores. Similar to the MRR score above expect measuring the probability
        that a DD-pairs with a positive clinical trials outcome will rank in the top k.


    Args:
        graph: Knowledge graph.
        model: Model making the predictions.
        model_name: Name of the model.
        transformers: Dictionary of trained transformers.
        cleaned_clinical_trial_data: Clinical trial data.
        train_data: Drug-disease pairs used for training the model.
        k_lst: List of integers for hit@k metrics.
        categ_pos: Column name for clinical trial category.
        interested_drug_nodes: DataFrame containing the drug nodes to consider in the evaluation.

    Returns:
        A list containing the AUROC, AP, MRR and Hit@k scores, a list of metric names, and the model name.
    """
    # Restricting and preparing drug-disease pairs
    if interested_drug_nodes:
        approved_drug_node_list = interested_drug_nodes[0].tolist()
    else:
        approved_drug_node_list = graph._drug_nodes

    # Extract drug-disease pairs with better clinical trials outcomes
    cleaned_clinical_trial_data_pos = cleaned_clinical_trial_data[
        cleaned_clinical_trial_data[categ_pos] == 1
    ]

    # Filter out rows with drugs not in approved_drug_node_list
    cleaned_clinical_trial_data_pos_filtered = cleaned_clinical_trial_data_pos[
        cleaned_clinical_trial_data_pos["drug_kg_id"].isin(approved_drug_node_list)
    ]

    test_data = pd.DataFrame(
        {
            "drug_kg_id": cleaned_clinical_trial_data_pos_filtered[
                "drug_kg_id"
            ].to_list(),
            "disease_kg_id": cleaned_clinical_trial_data_pos_filtered[
                "disease_kg_id"
            ].to_list(),
        }
    )

    result = perform_disease_centric_evaluation(
        graph,
        model,
        transformers,
        test_data,
        train_data,
        approved_drug_node_list,
        k_lst,
        is_return_curves=False,
    )
    result = list(result)
    return_result_name = ["AUROC", "AP", "MRR"] + [f"Hit@{k}" for k in k_lst]

    return (result, return_result_name, model_name)


def create_metrics_report(list_model_metrics_info) -> pd.DataFrame:
    """Function to consolidate metrics reports into master report.

    Args:
        metrics_reports: tuples of (name, report) pairs.

    Returns:
        DataFrame representing consolidated report.
    """
    # re-orgnize the info
    result = list(zip(*list_model_metrics_info))
    val = result[0]
    col = result[1]
    row = result[2]

    # create the dataframe
    report = pd.DataFrame(val, columns=col)
    report.index = row

    return report


def generator_example(generator):
    generator.generate(pd.DataFrame())
