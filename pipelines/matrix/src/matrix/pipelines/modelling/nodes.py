from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import numpy as np
import json
import pyspark.sql.functions as f

from pyspark.sql import DataFrame

from sklearn.model_selection import BaseCrossValidator
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt

from functools import wraps
from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import make_list_regexable

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from .model import ModelWrapper

plt.switch_backend("Agg")


def no_nulls(columns: List[str]):
    """Decorator to check columns for null values.

    FUTURE: Move to pandera when we figure out how to push messages for breaking changes.

    Args:
        columns: list of columns to check
    """

    if isinstance(columns, str):
        columns = [columns]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Proceed with the function if no null values are found
            df = func(*args, **kwargs)

            if not all(col_name in df.columns for col_name in columns):
                raise ValueError(f"DataFrame is missing required columns: {', '.join(columns)}")

            # Check if the specified column has any null values
            null_rows_df = df.filter(" OR ".join([f"{col_name} IS NULL" for col_name in columns]))

            # Check if the resulting DataFrame is empty
            if not null_rows_df.isEmpty():
                # Show rows with null values for debugging
                null_rows_df.show()
                raise ValueError(f"DataFrame contains null values in columns: {', '.join(columns)}")

            return df

        return wrapper

    return decorator


@has_schema(
    schema={"y": "int"},
    allow_subset=True,
)
@primary_key(primary_key=["source", "target"])
@no_nulls(columns=["source_embedding", "target_embedding"])
def create_int_pairs(
    nodes: DataFrame,
    raw_tp: DataFrame,
    raw_tn: DataFrame,
):
    """Create intermediate pairs dataset.

    Args:
        nodes: nodes dataframe
        raw_tp: Raw ground truth positive data.
        raw_tn: Raw ground truth negative data.

    Returns:
        Combined ground truth positive and negative data.
    """

    return (
        raw_tp.withColumn("y", f.lit(1))
        .unionByName(raw_tn.withColumn("y", f.lit(0)))
        .alias("pairs")
        .join(nodes.withColumn("source", f.col("id")), how="left", on="source")
        .withColumnRenamed("topological_embedding", "source_embedding")
        .join(nodes.withColumn("target", f.col("id")), how="left", on="target")
        .withColumnRenamed("topological_embedding", "target_embedding")
        .select("pairs.*", "source_embedding", "target_embedding")
    )


@has_schema(
    schema={
        "id": "string",
        "is_drug": "boolean",
        "is_disease": "boolean",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def prefilter_nodes(
    nodes: DataFrame, gt_pos: pd.DataFrame, drug_types: List[str], disease_types: List[str]
) -> DataFrame:
    """Prefilter nodes for negative sampling.

    Args:
        nodes: the nodes dataframe to be filtered
        gt_pos: dataframe with ground truth positives
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        Filtered nodes dataframe
    """

    ground_truth_nodes = (
        gt_pos.withColumn("id", f.col("source"))
        .unionByName(gt_pos.withColumn("id", f.col("target")))
        .select("id")
        .distinct()
        .withColumn("is_ground_pos", f.lit(True))
    )

    return (
        nodes.alias("nodes")
        .filter((f.col("category").isin(drug_types)) | (f.col("category").isin(disease_types)))
        .select("id", "category", "topological_embedding")
        .withColumn("is_drug", f.col("category").isin(drug_types))
        .withColumn("is_disease", f.col("category").isin(disease_types))
        .join(ground_truth_nodes, on="id", how="left")
        .fillna({"is_ground_pos": False})
    )


def apply_splitter(data: DataFrame, splitter: BaseCrossValidator) -> pd.DataFrame:
    """Apply a splitter to a dataframe.

    Args:
        data: Data to split.
        splitter: sklearn splitter object (BaseCrossValidator or its subclasses).
    Returns:
        Dataframe with additional columns.
    """
    all_data_frames = []
    for iteration, (train_index, test_index) in enumerate(splitter.split(data, data["y"])):
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
def make_splits(
    data: DataFrame,
    splitter: BaseCrossValidator,
) -> pd.DataFrame:
    """Function to split data and add columns containing the embeddings.

    Args:
        kg: kg dataset with nodes
        data: Data to split.
        splitter: sklearn splitter object (BaseCrossValidator or its subclasses).

    Returns:
        Data with split information.
    """
    all_data_frames = []
    for iteration, (train_index, test_index) in enumerate(splitter.split(data, data["y"])):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[:, "iteration"] = iteration
        fold_data.loc[train_index, "split"] = "TRAIN"
        fold_data.loc[test_index, "split"] = "TEST"
        all_data_frames.append(fold_data)

    return apply_splitter(data, splitter)


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
    generator: SingleLabelPairGenerator,
) -> pd.DataFrame:
    """Function to enrich the splits with drug-disease pairs.

    The generator is used to enrich the dataset with unknown drug-disease
    pairs. If a `IterativeDrugDiseasePair` generator is provided, the splits
    dataset is replicated.

    Args:
        graph: Knowledge graph.
        splits: Data splits.
        generator: SingleLabelPairGenerator instance.

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
    target_data = data.loc[mask, target_col_name] if target_col_name is not None else None

    # Iterate transformers
    fitted_transformers = {}
    for name, transform in transformers.items():
        # Fit transformer
        features = transform["features"]

        transformer = transform["transformer"].fit(data.loc[mask, features], target_data)

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
    """Function to run model class predictions on input data.

    Args:
        data: Data to predict on.
        model: Model making the predictions.
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
