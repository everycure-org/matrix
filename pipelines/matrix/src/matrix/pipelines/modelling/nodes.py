"""Module with nodes for modelling."""
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import numpy as np
import json
import bisect

from sklearn.model_selection._split import _BaseKFold
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from skopt.plots import plot_convergence

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph, DrugDiseasePairGenerator
from matrix.datasets.drp_model import DRPmodel, DRPmodel3classScikit



@has_schema(
    schema={
        "is_drug": "bool",
        "is_disease": "bool",
        "is_fda_approved": "bool",
        "embedding": "object",
    },
    allow_subset=True,
)
def create_feat_nodes(
    raw_nodes: pd.DataFrame,
    embeddings: pd.DataFrame,
    drug_types: List[str],
    disease_types: List[str],
    fda_list: List[str],
) -> pd.DataFrame:
    """Add features for nodes.

    Args:
        raw_nodes: Raw nodes data.
        embeddings: Embeddings data.
        drug_types: List of drug types.
        disease_types: List of disease types.
        fda_list: List of FDA approved drugs.

    Returns:
        Nodes enriched with features.
    """
    # Merge embeddings
    raw_nodes = raw_nodes.merge(embeddings, on="id", how="left")

    # Add flags
    raw_nodes["is_drug"] = raw_nodes["category"].apply(lambda x: x in drug_types)
    raw_nodes["is_disease"] = raw_nodes["category"].apply(lambda x: x in disease_types)
    raw_nodes["is_fda_approved"] = raw_nodes["id"].apply(lambda x: x in fda_list)

    return raw_nodes


@has_schema(
    schema={
        "source": "object",
        "source_embedding": "object",
        "target": "object",
        "target_embedding": "object",
        "y": "numeric",
    },
    allow_subset=True,
)
def create_prm_pairs(
    graph: KnowledgeGraph, raw_tp: pd.DataFrame, raw_tn: pd.DataFrame
) -> pd.DataFrame:
    """Create primary pairs dataset.

    Args:
        graph: Knowledge graph.
        raw_tp: Raw true positive data.
        raw_tn: Raw true negative data.

    Returns:
        Primary pairs dataset.
    """
    # Add label
    raw_tp["y"] = 1
    raw_tn["y"] = 0

    # Concat
    result = pd.concat([raw_tp, raw_tn], axis=0).reset_index(drop=True)

    # Add embeddings
    result["source_embedding"] = result.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    result["target_embedding"] = result.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )

    # Return concatenated data
    return result


@inject_object()
def make_splits(
    data: pd.DataFrame,
    splitter: _BaseKFold,
) -> pd.DataFrame:
    """Function to split data.

    Args:
        data: Data to split.
        splitter: sklearn splitter object.

    Returns:
        Data with split information.
    """
    all_data_frames = []
    for iteration, (train_index, test_index) in enumerate(
        splitter.split(data, data["y"])
    ):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[:, "iteration"] = iteration
        fold_data.loc[train_index, "split"] = "TRAIN"
        fold_data.loc[test_index, "split"] = "TEST"
        all_data_frames.append(fold_data)

    return pd.concat(all_data_frames, axis=0, ignore_index=True)


@has_schema(
    schema={
        "source": "object",
        "source_embedding": "object",
        "target": "object",
        "target_embedding": "object",
    },
    allow_subset=True,
)
@inject_object()
def create_model_input_nodes(
    graph: KnowledgeGraph,
    splits: pd.DataFrame,
    generator: DrugDiseasePairGenerator,
) -> pd.DataFrame:
    """Function to enrich the splits with drug-disease pairs.

    Args:
        graph: Knowledge graph.
        splits: Data splits.
        generator: DrugDiseasePairGenerator instance.

    Returns:
        Data with enriched splits.
    """
    generated = generator.generate(graph, splits)
    generated["split"] = "TRAIN"

    return pd.concat([splits, generated], axis=0, ignore_index=True)


@inject_object()
def apply_transformers(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    target_col_name: str = None,
) -> pd.DataFrame:
    """Function to apply a set of sklearn compatible transformers to the data.

    Args:
        data: Data to transform.
        transformers: Dictionary of transformers.
        target_col_name: Target column name.

    Returns:
        Transformed data.
    """
    # Ensure transformer only applied to train data
    mask = data["split"].eq("TRAIN")

    # Grab target data
    target_data = (
        data.loc[mask, target_col_name] if target_col_name is not None else None
    )
    
    # Iterate transformers
    for name, transform in transformers.items():
        # Fit transformer
        features = transform["features"]
        fitted_transformer = transform["transformer"].fit(
            data.loc[mask, features], target_data
        )

        # Updating dictionary of transformers
        transformers[name]["transformer"] = fitted_transformer

        # Apply transformer
        features_selected = data[features]
        transformed = pd.DataFrame(
            fitted_transformer.transform(features_selected),
            index=features_selected.index,
            columns=fitted_transformer.get_feature_names_out(features_selected),
        )

        # Overwrite columns
        data = pd.concat(
            [data.drop(columns=features), transformed],
            axis="columns",
        )
        
    return data, transformers


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def tune_parameters(
    data: pd.DataFrame,
    tuner: Any,
    features: List[str],
    target_col_name: str,
    enable_regex: str = True,
) -> Tuple[Dict,]:
    """Function to apply hyperparameter tuning.

    Args:
        data: Data to tune on.
        tuner: Tuner object.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
        enable_regex: Enable regex for features.

    Returns:
        Refit compatible dictionary of best parameters.
    """
    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    # Fit tuner
    result = tuner.fit(X_train.values, y_train.values)

    return json.loads(
        json.dumps(
            {
                "object": f"{type(tuner._estimator).__module__}.{type(tuner._estimator).__name__}",
                **tuner.best_params_,
            },
            default=int,
        )
    ), plot_convergence(result).figure


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def train_model(
    data: pd.DataFrame,
    estimator: BaseEstimator,
    features: List[str],
    target_col_name: str,
    enable_regex: bool = True,
) -> Dict:
    """Function to train model on the given data.

    Args:
        data: Data to train on.
        estimator: sklearn estimator.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
        enable_regex: Enable regex for features.

    Returns:
        Trained model.
    """
    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    return estimator.fit(X_train.values, y_train.values)


def generate_drp_model(
        estimator : BaseEstimator,
        graph: KnowledgeGraph, 
        transformers : Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
        features : List[str]
) -> DRPmodel3classScikit:
    """Returns instance of the class DRPmodel3classScikit.

    Args:
        estimator (BaseEstimator): sklearn estimator.
        graph (KnowledgeGraph): List of features, may be regex specified.
        transformers (Dict[str, Dict[str, Union[_BaseImputer, List[str]]]]):
            Dictionary of fitted transformers.
        features (List[str]): List of features, may be regex specified.
    """
    return DRPmodel3classScikit(estimator, graph, transformers, features)


@inject_object()
def get_classification_metrics(
    drp_model: DRPmodel,
    data: pd.DataFrame,
    metrics: List[callable],
    target_col_name: str,
) -> Dict:
    """Function to evaluate model performance.
    TO DO: modify to include AUROC classification score

    Args:
        drp_model (DRPmodel): Model giving a probability score.
        data (pd.DataFrame): Data to evaluate.
        metrics (List[callable]): List of callable metrics.
        target_col_name (str): Target column name.

    Returns:
        Dictionary containing report
    """
    report = {}

    for metric in metrics:
        for split in ["TEST", "TRAIN"]:
            split_index = data["split"].eq(split)
            y_true = data.loc[split_index, target_col_name]
            treat_scores = drp_model.give_treat_scores(data[split_index], skip_vectorise = True)['treat score']

            # Execute metric
            report[f"{split.lower()}_{metric.__name__}"] = metric(
                y_true == 1, treat_scores > 0.5
            ).item()

    return json.loads(json.dumps(report))



def perform_disease_centric_evaluation(
    drp_model: DRPmodel,
    known_data : pd.DataFrame, 
    k_lst : List[int] = [2,10], # TO DO: put into params file 
) -> Dict:
    """
    !!! This version will use drugs with known positives
    TO DO: docstring using the time-split analysis notebook
    """
    # Extracting known positive test data and training data
    is_test = known_data["split"].eq("TEST")
    is_pos = known_data["y"].eq(1)
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
        all_pos_test = kp_data_test[kp_data_test["target"]==disease]
        all_train = train_data[train_data["target"]==disease]

        # Construct negatives
        check_cond_pos_test =  lambda drug: drug not in list(all_pos_test["source"])
        check_cond_train =  lambda drug: drug not in list(all_train["source"])
        check_conds = lambda drug: check_cond_pos_test(drug) and check_cond_train(drug)
        negative_drugs = all_drugs[all_drugs.map(check_conds)]
        negative_pairs = pd.DataFrame({"source":negative_drugs, "target":disease})
        negative_pairs['y'] = 0

        # Compute probability scores 
        all_pos_test = drp_model.give_treat_scores(all_pos_test, skip_vectorise=True)
        negative_pairs = drp_model.give_treat_scores(negative_pairs)

        # Concatenate to DataFrame with all probability scores
        if df_all_exists:
            df_all = pd.concat((df_all, all_pos_test, negative_pairs), ignore_index=True)
        else: 
            df_all = pd.concat((all_pos_test, negative_pairs), ignore_index=True)
            df_all_exists = True

        # Compute rank for all positives
        negative_pairs = negative_pairs.sort_values("treat score", ascending = True)
        for _, prob in all_pos_test["treat score"].items():
            rank = len(negative_pairs) - bisect.bisect_left(list(negative_pairs["treat score"]), prob) + 1
            # Add to total
            mrr_total += 1/rank
            for idx, k in enumerate(k_lst):
                if rank <= k:
                    hitk_total_lst[idx] += 1

    mrr = mrr_total/len(kp_data_test)
    hitk_lst = list(hitk_total_lst/len(kp_data_test))

    # Computing AUROC and AP
    y_true = df_all["y"]
    y_score = df_all["treat score"]
    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    report = {"AUROC": auroc, "AP": ap, "MRR": mrr, "Hit@k": hitk_lst} # TO DO: fix format
    return json.loads(json.dumps(report))
