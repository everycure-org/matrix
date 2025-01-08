import logging
from typing import Any, Callable, Dict, List, Union, Tuple
import pandas as pd
from pyspark.sql import DataFrame
import json
from pandera.typing import Series
from pandera.pyspark import DataFrameModel as PysparkDataFrameModel
from pandera import DataFrameModel as PandasDataFrameModel
import pandera
from pyspark.sql import functions as f
import pyspark.sql.types as T
from pandera import Field as PandasField


from sklearn.model_selection import BaseCrossValidator
from sklearn.impute._base import _BaseImputer
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt

from functools import wraps
from matrix.inject import inject_object, make_list_regexable, unpack_params

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from .model import ModelWrapper

logger = logging.getLogger(__name__)

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


def filter_valid_pairs(
    nodes: DataFrame,
    edges_gt: DataFrame,
    drug_categories: List[str],
    disease_categories: List[str],
) -> Tuple[DataFrame, Dict[str, float]]:
    """Filter GT pairs to only include nodes that 1) exist in the nodes DataFrame, 2) have the correct category.

    Args:
        nodes: Nodes dataframe
        eges_gt: DataFrame with ground truth pairs
        drug_categories: List of drug categories to be filtered on
        disease_categories: List of disease categories to be filtered on

    Returns:
        Tuple containing:
        - DataFrame with combined filtered positive and negative pairs
        - Dictionary with retention statistics
    """
    # Create set of categories to filter on
    categories = drug_categories + disease_categories

    # Get list of nodes in the KG
    valid_nodes_in_kg = nodes.select("id").distinct()
    valid_nodes_with_categories = nodes.filter(f.col("category").isin(categories)).select("id")

    # Divide edges_gt into positive and negative pairs to know ratio retained for each
    edges_gt = edges_gt.withColumnRenamed("subject", "source").withColumnRenamed("object", "target")
    raw_tp = edges_gt.filter(f.col("y") == 1)
    raw_tn = edges_gt.filter(f.col("y") == 0)

    # Filter out pairs where both source and target exist in nodes
    filtered_tp_in_kg = (
        raw_tp.join(valid_nodes_in_kg.alias("source_nodes"), raw_tp.source == f.col("source_nodes.id"))
        .join(valid_nodes_in_kg.alias("target_nodes"), raw_tp.target == f.col("target_nodes.id"))
        .select(raw_tp["*"])
    )
    filtered_tn_in_kg = (
        raw_tn.join(valid_nodes_in_kg.alias("source_nodes"), raw_tn.source == f.col("source_nodes.id"))
        .join(valid_nodes_in_kg.alias("target_nodes"), raw_tn.target == f.col("target_nodes.id"))
        .select(raw_tn["*"])
    )
    filtered_tp_categories = (
        raw_tp.join(valid_nodes_with_categories.alias("source_nodes"), raw_tp.source == f.col("source_nodes.id"))
        .join(valid_nodes_with_categories.alias("target_nodes"), raw_tp.target == f.col("target_nodes.id"))
        .select(raw_tp["*"])
    )
    filtered_tn_categories = (
        raw_tn.join(valid_nodes_with_categories.alias("source_nodes"), raw_tn.source == f.col("source_nodes.id"))
        .join(valid_nodes_with_categories.alias("target_nodes"), raw_tn.target == f.col("target_nodes.id"))
        .select(raw_tn["*"])
    )
    # Filter out pairs where category of source or target is incorrect AND source and target do not exist in nodes
    final_filtered_tp_categories = (
        filtered_tp_in_kg.join(
            valid_nodes_with_categories.alias("source_nodes"), filtered_tp_categories.source == f.col("source_nodes.id")
        )
        .join(
            valid_nodes_with_categories.alias("target_nodes"), filtered_tp_categories.target == f.col("target_nodes.id")
        )
        .select(filtered_tp_categories["*"])
    )
    final_filtered_tn_categories = (
        filtered_tn_in_kg.join(
            valid_nodes_with_categories.alias("source_nodes"), filtered_tn_categories.source == f.col("source_nodes.id")
        )
        .join(
            valid_nodes_with_categories.alias("target_nodes"), filtered_tn_categories.target == f.col("target_nodes.id")
        )
        .select(filtered_tn_categories["*"])
    )
    # Calculate retention percentages
    retention_stats = {
        "positive_pairs_retained_in_kg_pct": (filtered_tp_in_kg.count() / raw_tp.count())
        if raw_tp.count() > 0
        else 1.0,
        "negative_pairs_retained_in_kg_pct": (filtered_tn_in_kg.count() / raw_tn.count())
        if raw_tn.count() > 0
        else 1.0,
        "positive_pairs_retained_in_categories_pct": (filtered_tp_categories.count() / raw_tp.count())
        if raw_tp.count() > 0
        else 1.0,
        "negative_pairs_retained_in_categories_pct": (filtered_tn_categories.count() / raw_tn.count())
        if raw_tn.count() > 0
        else 1.0,
        "positive_pairs_retained_final_pct": (final_filtered_tp_categories.count() / raw_tp.count())
        if raw_tp.count() > 0
        else 1.0,
        "negative_pairs_retained_final_pct": (final_filtered_tn_categories.count() / raw_tn.count())
        if raw_tn.count() > 0
        else 1.0,
    }

    # Combine filtered pairs
    pairs_df = final_filtered_tp_categories.withColumn("y", f.lit(1)).unionByName(
        final_filtered_tn_categories.withColumn("y", f.lit(0))
    )
    return {"pairs": pairs_df, "metrics": retention_stats}


class EmbeddingsWithPairsSchema(PysparkDataFrameModel):
    y: T.IntegerType
    source_embedding: T.ArrayType(T.FloatType())  # type: ignore
    target_embedding: T.ArrayType(T.FloatType())  # type: ignore

    class Config:
        strict = False


@pandera.check_output(EmbeddingsWithPairsSchema)
def attach_embeddings(
    pairs_df: DataFrame,
    nodes: DataFrame,
) -> DataFrame:
    """Attach node embeddings to the pairs DataFrame.

    Args:
        pairs_df: DataFrame containing source-target pairs
        nodes: nodes dataframe containing embeddings

    Returns:
        DataFrame with source and target embeddings attached
    """
    return (
        pairs_df.alias("pairs")
        .join(nodes.withColumn("source", f.col("id")), how="left", on="source")
        .withColumnRenamed("topological_embedding", "source_embedding")
        .join(nodes.withColumn("target", f.col("id")), how="left", on="target")
        .withColumnRenamed("topological_embedding", "target_embedding")
        .select("pairs.*", "source_embedding", "target_embedding")
    )


class NodeSchema(PysparkDataFrameModel):
    id: T.StringType
    is_drug: T.BooleanType
    is_disease: T.BooleanType

    class Config:
        strict = False
        unique = ["id"]


@pandera.check_output(NodeSchema)
def prefilter_nodes(
    full_nodes: DataFrame,
    nodes: DataFrame,
    gt: DataFrame,
    drug_types: List[str],
    disease_types: List[str],
) -> DataFrame:
    """Prefilter nodes for negative sampling.

    Args:
        nodes: the nodes dataframe to be filtered
        gt: dataframe with ground truth positives and negatives
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        Filtered nodes dataframe
    """
    gt_pos = gt.filter(f.col("y") == 1)
    ground_truth_nodes = (
        gt.withColumn("id", f.col("source"))
        .unionByName(gt_pos.withColumn("id", f.col("target")))
        .select("id")
        .distinct()
        .withColumn("in_ground_pos", f.lit(True))
    )

    df = (
        nodes.withColumn("is_drug", f.arrays_overlap(f.col("all_categories"), f.lit(drug_types)))
        .withColumn("is_disease", f.arrays_overlap(f.col("all_categories"), f.lit(disease_types)))
        .filter((f.col("is_disease")) | (f.col("is_drug")))
        .select("id", "topological_embedding", "is_drug", "is_disease")
        # TODO: The integrated data product _should_ contain these nodes
        # TODO: Verify below does not have any undesired side effects
        .join(ground_truth_nodes, on="id", how="left")
        .fillna({"in_ground_pos": False})
    )
    return df


@inject_object()
def make_folds(
    data: DataFrame,
    splitter: BaseCrossValidator,
) -> Tuple[pd.DataFrame]:
    """Function to generate folds for modelling.

    NOTE: This currently loads the `n_splits` from the settings, as this
    pipeline is generated dynamically to allow parallelization.

         _______
       .-       -.
      /           \
     |,  .-.  .-.  ,|
     | )(_o/  \o_)( |
     |/     /\     \|
     (_     ^^     _)
      \__|IIIIII|__/
       | \IIIIII/ |
       \          /
        `--------`

    Args:
        data: dataframe
        splitter: splitter
    Returns:
        Tuple of dataframes with data for each fold, dfs 1-k are 
        dfs with data for folds, df k+1 is training data only.
    """
    # Set number of splits
    all_data_frames = make_splits(data, splitter)

    # Add "training data only" fold
    full_data = data.copy()
    full_data.loc[:, "split"] = "TRAIN"
    return all_data_frames + tuple([full_data])


@inject_object()
def make_splits(
    data: DataFrame,
    splitter: BaseCrossValidator,
) -> Tuple[pd.DataFrame]:
    """Function to split data.

    FUTURE: Update to produce single DF only, where we add a column identifying the fold.

    Args:
        kg: kg dataset with nodes
        data: Data to split.
        splitter: sklearn splitter object (BaseCrossValidator or its subclasses).
        n_splits: number of splits
    Returns:
        Tuple of dataframes for each fold.
    """

    # Split data into folds
    all_data_frames = []
    for train_index, test_index in splitter.split(data, data["y"]):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[train_index, "split"] = "TRAIN"
        fold_data.loc[test_index, "split"] = "TEST"
        all_data_frames.append(fold_data)

    return tuple(all_data_frames)


class ModelSplitsSchema(PandasDataFrameModel):
    source: Series[object] = PandasField(nullable=True)
    source_embedding: Series[object] = PandasField(nullable=True)
    target: Series[object] = PandasField(nullable=True)
    target_embedding: Series[object] = PandasField(nullable=True)
    split: Series[object] = PandasField(nullable=True)

    class Config:
        strict = False


@pandera.check_output(ModelSplitsSchema)
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
@make_list_regexable(source_df="data", make_regexable_kwarg="features")
def tune_parameters(
    data: pd.DataFrame,
    tuner: Any,
    features: List[str],
    target_col_name: str,
) -> Tuple[Dict,]:
    """Function to apply hyperparameter tuning.

    Args:
        data: Data to tune on.
        tuner: Tuner object.
        features: List of features, may be regex specified.
        target_col_name: Target column name.

    Returns:
        Refit compatible dictionary of best parameters.
    """
    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    # Fit tuner
    tuner.fit(X_train.values, y_train.values)

    estimator = getattr(tuner, "estimator", None)
    if estimator is None:
        raise ValueError("Tuner must have 'estimator' attribute")

    return json.loads(
        json.dumps(
            {
                "object": f"{type(estimator).__module__}.{type(estimator).__name__}",
                **tuner.best_params_,
            },
            default=int,
        )
    ), tuner.convergence_plot if hasattr(tuner, "convergence_plot") else plt.figure()


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable_kwarg="features")
def train_model(
    data: pd.DataFrame,
    estimator: BaseEstimator,
    features: List[str],
    target_col_name: str,
) -> Dict:
    """Function to train model on the given data.

    Args:
        data: Data to train on.
        estimator: sklearn compatible estimator.
        features: List of features, may be regex specified.
        target_col_name: Target column name.

    Returns:
        Trained model.
    """
    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    logger.info(f"Starting model: {estimator} training...")
    estimator_fit = estimator.fit(X_train.values, y_train.values)
    logger.info("Model training completed...")
    return estimator_fit


@inject_object()
def create_model(agg_func: Callable, *estimators) -> ModelWrapper:
    """Function to create final model.

    Args:
        agg_func: function to aggregate ensemble models' treat score
        estimators: list of fitted estimators
    Returns:
        ModelWrapper encapsulating estimators
    """
    return ModelWrapper(estimators=estimators, agg_func=agg_func)


@inject_object()
@make_list_regexable(source_df="data", make_regexable_kwarg="features")
def get_model_predictions(
    data: pd.DataFrame,
    model: ModelWrapper,
    features: List[str],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> pd.DataFrame:
    """Function to run model class predictions on input data.

    Args:
        data: Data to predict on.
        model: Model making the predictions.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
        prediction_suffix: Suffix to add to the prediction column, defaults to '_pred'.

    Returns:
        Data with predictions.
    """
    data[target_col_name + prediction_suffix] = model.predict(data[features].values)
    return data


def combine_data(*predictions_all_folds: pd.DataFrame) -> pd.DataFrame:
    """Returns combined dataframe containing predictions from all folds.

    Args:
        data_all_folds: Dataframes containing model predictions for all folds.
    """
    return pd.concat(predictions_all_folds)


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

    # Return None for each metric if there is no test data
    if not data["split"].eq("TEST").any():
        return {name: None for name in metrics.keys()}

    # Compute evaluation metrics and add to report
    for name, func in metrics.items():
        for split in ["TEST", "TRAIN"]:
            split_index = data["split"].eq(split)
            y_true = data.loc[split_index, target_col_name]
            y_pred = data.loc[split_index, target_col_name + prediction_suffix]

            # Execute metric
            report[f"{split.lower()}_{name}"] = func(y_true, y_pred)

    return json.loads(json.dumps(report, default=float))


@inject_object()
def aggregate_metrics(aggregation_functions: List[Dict], *metrics) -> Dict:
    """
    Aggregate metrics for the separate folds into a single set of metrics.

    Args:
        aggregation_functions: List of dictionaries containing the name and object of the aggregation function.
        metrics: Dictionaries of metrics for all folds.
    """

    # Extract list of metrics for each fold and check consistency
    metric_names_lst_all_folds = [list(report.keys()) for report in metrics]
    metric_names_lst = metric_names_lst_all_folds[0]
    if not all(metric_names == metric_names_lst_all_folds[0] for metric_names in metric_names_lst_all_folds):
        raise ValueError("Inconsistent metrics across folds. Each fold should have the same set of metrics.")

    # Perform aggregation
    aggregated_metrics = dict()
    for agg_func in aggregation_functions:
        aggregated_metrics[agg_func.__name__] = {
            metric_name: agg_func([report[metric_name] for report in metrics]) for metric_name in metric_names_lst
        }

    return json.loads(json.dumps(aggregated_metrics, default=float))
