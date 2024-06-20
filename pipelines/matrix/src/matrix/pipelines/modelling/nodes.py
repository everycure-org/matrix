"""Module with nodes for modelling."""
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import numpy as np
import json

from pyspark.sql import DataFrame

from sklearn.model_selection._split import _BaseKFold
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph, DrugDiseasePairGenerator
from .model import ModelWrapper


@has_schema(
    schema={
        "is_drug": "bool",
        "is_disease": "bool",
        "is_ground_pos": "bool",
        "embedding": "object",
    },
    allow_subset=True,
)
def create_feat_nodes(
    raw_nodes: DataFrame,
    embeddings: pd.DataFrame,
    known_pairs: DataFrame,
    drug_types: List[str],
    disease_types: List[str],
) -> pd.DataFrame:
    """Add features for nodes.

    FUTURE: Add flags for official set of drugs and diseases when we have them.

    Args:
        raw_nodes: Raw nodes data.
        embeddings: Embeddings data.
        known_pairs: Ground truth data.
        drug_types: List of drug types.
        disease_types: List of disease types.

    Returns:
        Nodes enriched with features.
    """
    # Merge embeddings
    raw_nodes = raw_nodes.toPandas().merge(embeddings, on="id", how="left")

    # Add drugs and diseases types flags
    raw_nodes["is_drug"] = raw_nodes["category"].apply(lambda x: x in drug_types)
    raw_nodes["is_disease"] = raw_nodes["category"].apply(lambda x: x in disease_types)

    # Add flag for set of drugs appearing in ground truth positive set
    known_pairs = known_pairs.toPandas()
    ground_pos = known_pairs[known_pairs["y"].eq(1)]
    ground_pos_drug_ids = list(ground_pos["source"].unique())
    ground_pos_disease_ids = list(ground_pos["target"].unique())
    raw_nodes["is_ground_pos"] = raw_nodes["id"].isin(
        ground_pos_drug_ids + ground_pos_disease_ids
    )

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
    graph: KnowledgeGraph,
    data: DataFrame,
) -> pd.DataFrame:
    """Create primary pairs dataset.

    Args:
        graph: Knowledge graph.
        data: Pairs dataset to enrich with embeddings.

    Returns:
        Ground truth data enriched with embeddings.
    """
    # Add embeddings
    data = data.toPandas()
    data["source_embedding"] = data.apply(
        lambda row: graph._embeddings[row.source], axis=1
    )
    data["target_embedding"] = data.apply(
        lambda row: graph._embeddings[row.target], axis=1
    )

    # Return enriched data
    return data


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

    return pd.concat(all_data_frames, axis="index", ignore_index=True)


@has_schema(
    schema={
        "source": "object",
        "source_embedding": "object",
        "target": "object",
        "target_embedding": "object",
        "iteration": "numeric",
        "split": "object",
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

    The generator is used to enrich the dataset with unknown drug-disease
    pairs. If a `IterativeDrugDiseasePair` generator is provided, the splits
    dataset is replicated.

    Args:
        graph: Knowledge graph.
        splits: Data splits.
        generator: DrugDiseasePairGenerator instance.

    Returns:
        Data with enriched splits.
    """
    generated = generator.generate(graph, splits)
    generated["split"] = "TRAIN"

    return pd.concat([splits, generated], axis="index", ignore_index=True)


@inject_object()
def fit_transformers(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    target_col_name: str = None,
) -> pd.DataFrame:
    """Function fit transformers to the data.

    Args:
        data: Data to transform.
        transformers: Dictionary of transformers.
        target_col_name: Column name to predict.

    Returns:
        Fitted transformers.
    """
    # Ensure transformer only fit on training data
    mask = data["split"].eq("TRAIN")

    # Grab target data
    target_data = (
        data.loc[mask, target_col_name] if target_col_name is not None else None
    )

    # Iterate transformers
    fitted_transformers = {}
    for name, transform in transformers.items():
        # Fit transformer
        features = transform["features"]

        transformer = transform["transformer"].fit(
            data.loc[mask, features], target_data
        )

        fitted_transformers[name] = {"transformer": transformer, "features": features}

    return fitted_transformers


@inject_object()
def apply_transformers(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
) -> pd.DataFrame:
    """Function apply fitted transformers to the data.

    Args:
        data: Data to transform.
        transformers: Dictionary of transformers.

    Returns:
        Transformed data.
    """
    # Iterate transformers
    for _, transformer in transformers.items():
        # Apply transformer
        features = transformer["features"]
        features_selected = data[features]

        transformed = pd.DataFrame(
            transformer["transformer"].transform(features_selected),
            index=features_selected.index,
            columns=transformer["transformer"].get_feature_names_out(features_selected),
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
        estimator: sklearn compatible estimator.
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


def create_model(*estimators) -> ModelWrapper:
    """Function to create final model.

    Args:
        estimators: list of fitted estimators
    Returns:
        ModelWrapper encapsulating estimators
    """
    return ModelWrapper(estimators=estimators, agg_func=np.mean)


@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def get_model_predictions(
    data: pd.DataFrame,
    model: ModelWrapper,
    features: List[str],
    target_col_name: str,
    prediction_suffix: str = "_pred",
    enable_regex: str = True,
) -> pd.DataFrame:
    """Function to run model predictions on input data.

    Args:
        data: Data to predict on.
        model: model.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
        prediction_suffix: Suffix to add to the prediction column, defaults to '_pred'.
        enable_regex: Enable regex for features.

    Returns:
        Data with predictions.
    """
    data[target_col_name + prediction_suffix] = model.predict(data[features].values)
    return data


@inject_object()
def check_model_performance(
    data: pd.DataFrame,
    metrics: List[callable],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> Dict:
    """Function to evaluate model performance on the training data and ground truth test data.

    NOTE: This function only provides a partial indication of model performance,
    primarily for checking that a model has been successfully trained.

    Args:
        data: Data to evaluate.
        metrics: List of callable metrics.
        target_col_name: Target column name.
        prediction_suffix: Suffix to add to the prediction column, defaults to '_pred'.

    Returns:
        Dictionary containing report
    """
    report = {}

    for name, func in metrics.items():
        for split in ["TEST", "TRAIN"]:
            split_index = data["split"].eq(split)
            y_true = data.loc[split_index, target_col_name]
            y_pred = data.loc[split_index, target_col_name + prediction_suffix]

            # Execute metric
            report[f"{split.lower()}_{name}"] = func(y_true, y_pred)

    return json.loads(json.dumps(report, default=float))


def consolidate_reports(*reports) -> dict:
    """Function to consolidate reports into master report.

    Args:
        reports: tuples of (name, report) pairs.

    Returns:
        Dictionary representing consolidated report.
    """
    return [*reports]
