"""Module with nodes for modelling."""
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import json

from sklearn.model_selection._split import _BaseKFold
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph, DrugDiseasePairGenerator


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
    for _, transform in transformers.items():
        # Fit transformer
        features = transform["features"]
        transformer = transform["transformer"].fit(
            data.loc[mask, features], target_data
        )

        # Apply transformer
        features_selected = data[features]
        transformed = pd.DataFrame(
            transformer.transform(features_selected),
            index=features_selected.index,
            columns=transformer.get_feature_names_out(features_selected),
        )

        # Overwrite columns
        data = pd.concat(
            [data.drop(columns=features), transformed],
            axis="columns",
        )

    return data


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
    tuner.fit(X_train.values, y_train.values)

    return json.loads(
        json.dumps(
            {
                "object": f"{type(tuner._estimator).__module__}.{type(tuner._estimator).__name__}",
                **tuner.best_params_,
            },
            default=int,
        )
    ), tuner.convergence_plot if hasattr(tuner, "convergence_plot") else plt.figure()


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


@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def get_model_predictions(
    data: pd.DataFrame,
    estimator: BaseEstimator,
    features: List[str],
    target_col_name: str,
    prediction_suffix: str = "_pred",
    enable_regex: str = True,
) -> pd.DataFrame:
    """Function to run model predictions on input data.

    Args:
        data: Data to predict on.
        estimator: sklearn estimator.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
        prediction_suffix: Suffix to add to the prediction column, defaults to '_pred'.
        enable_regex: Enable regex for features.

    Returns:
        Data with predictions.
    """
    data[target_col_name + prediction_suffix] = estimator.predict(data[features].values)

    return data


@inject_object()
def get_model_performance(
    data: pd.DataFrame,
    metrics: List[callable],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> Dict:
    """Function to evaluate model performance.

    Args:
        data: Data to evaluate.
        metrics: List of callable metrics.
        target_col_name: Target column name.
        prediction_suffix: Suffix to add to the prediction column, defaults to '_pred'.

    Returns:
        Dictionary containing report
    """
    report = {}

    for metric in metrics:
        for split in ["TEST", "TRAIN"]:
            split_index = data["split"].eq(split)
            y_true = data.loc[split_index, target_col_name]
            y_prediction = data.loc[split_index, target_col_name + prediction_suffix]

            # Execute metric
            # FUTURE: This currently mergers the unknown and known classes
            report[f"{split.lower()}_{metric.__name__}"] = metric(
                y_true == 1, y_prediction == 1
            ).item()

    return json.loads(json.dumps(report))


def consolidate_reports(*reports) -> dict:
    """Function to consolidate reports into master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """
    print(reports)
