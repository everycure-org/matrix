import itertools
import json
import logging
from typing import Any, Callable, Iterable, Union

import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.types as T
from pyspark.sql import functions as f
from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import BaseCrossValidator

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import DiseaseSplitDrugDiseasePairGenerator, SingleLabelPairGenerator
from matrix.inject import OBJECT_KW, inject_object, make_list_regexable, unpack_params
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

from .model import ModelWrapper
from .model_selection import DiseaseAreaSplit

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


def filter_valid_pairs(
    nodes: ps.DataFrame,
    edges_gt: ps.DataFrame,
    drug_categories: Iterable[str],
    disease_categories: Iterable[str],
) -> tuple[ps.DataFrame, dict[str, float]]:
    """Filter GT pairs to only include nodes that 1) exist in the nodes DataFrame, 2) have the correct category.

    Args:
        nodes: Nodes dataframe
        edges_gt: DataFrame with ground truth pairs
        drug_categories: list of drug categories to be filtered on
        disease_categories: list of disease categories to be filtered on

    Returns:
        tuple containing:
        - DataFrame with combined filtered positive and negative pairs
        - Dictionary with retention statistics
    """
    # Create set of categories to filter on
    categories = set(itertools.chain(drug_categories, disease_categories))
    categories_array = f.array([f.lit(cat) for cat in categories])

    # Get list of nodes in the KG
    valid_nodes_in_kg = nodes.select("id").distinct().cache()
    valid_nodes_with_categories = (
        nodes.filter(f.size(f.array_intersect(f.col("all_categories"), categories_array)) > 0).select("id").cache()
    )
    # Divide edges_gt into positive and negative pairs to know ratio retained for each
    edges_gt = edges_gt.withColumnRenamed("subject", "source").withColumnRenamed("object", "target")
    raw_tp = edges_gt.filter(f.col("y") == 1).cache()
    raw_tn = edges_gt.filter(f.col("y") == 0).cache()

    # Filter out pairs where both source and target exist in nodes
    filtered_tp_in_kg = _filter_source_and_target_exist(raw_tp, in_=valid_nodes_in_kg)
    filtered_tn_in_kg = _filter_source_and_target_exist(raw_tn, in_=valid_nodes_in_kg)
    filtered_tp_categories = _filter_source_and_target_exist(raw_tp, in_=valid_nodes_with_categories)
    filtered_tn_categories = _filter_source_and_target_exist(raw_tn, in_=valid_nodes_with_categories)

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
    rows_in_raw_tp, rows_in_raw_tn = raw_tp.count(), raw_tn.count()
    retention_stats = {
        "positive_pairs_retained_in_kg_pct": (filtered_tp_in_kg.count() / rows_in_raw_tp) if rows_in_raw_tp else 1.0,
        "negative_pairs_retained_in_kg_pct": (filtered_tn_in_kg.count() / rows_in_raw_tn) if rows_in_raw_tn else 1.0,
        "positive_pairs_retained_in_categories_pct": (filtered_tp_categories.count() / rows_in_raw_tp)
        if rows_in_raw_tp
        else 1.0,
        "negative_pairs_retained_in_categories_pct": (filtered_tn_categories.count() / rows_in_raw_tn)
        if rows_in_raw_tn
        else 1.0,
        "positive_pairs_retained_final_pct": (final_filtered_tp_categories.count() / rows_in_raw_tp)
        if rows_in_raw_tp
        else 1.0,
        "negative_pairs_retained_final_pct": (final_filtered_tn_categories.count() / rows_in_raw_tn)
        if rows_in_raw_tn
        else 1.0,
    }

    # Combine filtered pairs
    pairs_df = final_filtered_tp_categories.withColumn("y", f.lit(1)).unionByName(
        final_filtered_tn_categories.withColumn("y", f.lit(0))
    )
    return {"pairs": pairs_df, "metrics": retention_stats}


def _filter_source_and_target_exist(df: ps.DataFrame, in_: ps.DataFrame) -> ps.DataFrame:
    return (
        df.join(in_.alias("source_nodes"), df["source"] == f.col("source_nodes.id"))
        .join(in_.alias("target_nodes"), df["target"] == f.col("target_nodes.id"))
        .select(df["*"])
    )


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(T.StringType(), nullable=False),
            "target": Column(T.StringType(), nullable=False),
            "source_embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
            "target_embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
        },
        unique=["source", "target"],
    )
)
def attach_embeddings(
    pairs_df: ps.DataFrame,
    nodes: ps.DataFrame,
) -> ps.DataFrame:
    """Attach node embeddings to the pairs DataFrame.

    Args:
        pairs_df: DataFrame containing source-target pairs
        nodes: nodes dataframe containing embeddings

    Returns:
        DataFrame with source and target embeddings attached
    """
    return pairs_df.transform(_add_embedding, from_=nodes, using="source").transform(
        _add_embedding, from_=nodes, using="target"
    )


def _add_embedding(df: ps.DataFrame, from_: ps.DataFrame, using: str) -> ps.DataFrame:
    from_ = from_.select(
        f.col("id").alias(using),
        f.col("topological_embedding").cast(T.ArrayType(T.FloatType())).alias(f"{using}_embedding"),
    )
    return df.join(from_, how="left", on=using)


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "is_drug": Column(T.BooleanType(), nullable=False),
            "is_disease": Column(T.BooleanType(), nullable=False),
        },
        unique=["id"],
    )
)
def prefilter_nodes(
    nodes: ps.DataFrame,
    gt: ps.DataFrame,
    drug_types: list[str],
    disease_types: list[str],
) -> ps.DataFrame:
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
        .filter(f.col("is_disease") | f.col("is_drug"))
        .select("id", "topological_embedding", "is_drug", "is_disease")
        # TODO: The integrated data product _should_ contain these nodes
        # TODO: Verify below does not have any undesired side effects
        .join(ground_truth_nodes, on="id", how="left")
        .fillna({"in_ground_pos": False})
    )
    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            "source_embedding": Column(object, nullable=False),
            "target": Column(str, nullable=False),
            "target_embedding": Column(object, nullable=False),
            "split": Column(str, nullable=False),
            "fold": Column(int, nullable=False),
        },
        unique=["fold", "source", "target"],
    )
)
@inject_object()
def make_folds(
    data: pd.DataFrame,
    splitter: BaseCrossValidator,
    disease_list: pd.DataFrame = None,
) -> pd.DataFrame:
    """Function to split data.

    Args:
        data: Data to split.
        splitter: sklearn splitter object (BaseCrossValidator or its subclasses).
        disease_list: disease list from https://github.com/everycure-org/matrix-disease-list/.
            Required only when using DiseaseAreaSplit.

    Returns:
        Dataframe with test-train split for all folds.
        By convention, folds 0 to k-1 are the proper test train splits for k-fold cross-validation,
        while fold k is the fold with full training data
    """

    # Split data into folds
    all_data_frames = []
    # FUTURE: Ensure fields are reflected in GT dataset for future splitters
    if isinstance(splitter, DiseaseAreaSplit):
        if disease_list is None:
            raise ValueError("disease_list is required when using DiseaseAreaSplit")
        split_iterator = splitter.split(data, disease_list)
    else:
        split_iterator = splitter.split(data, data["y"])

    for fold, (train_index, test_index) in enumerate(split_iterator):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[train_index, "split"] = "TRAIN"
        fold_data.loc[test_index, "split"] = "TEST"
        fold_data.loc[:, "fold"] = fold
        all_data_frames.append(fold_data)

    # Add fold for full training data
    full_fold_data = data.copy()
    full_fold_data["split"] = "TRAIN"
    full_fold_data["fold"] = splitter.n_splits
    all_data_frames.append(full_fold_data)

    return pd.concat(all_data_frames, ignore_index=True)


@inject_object()
@check_output(
    schema=DataFrameSchema(
        columns={
            "source": Column(str, nullable=False),
            "source_embedding": Column(object, nullable=False),
            "target": Column(str, nullable=False),
            "target_embedding": Column(object, nullable=False),
            "split": Column(str, nullable=False),
            "fold": Column(int, nullable=False),
        },
        # unique=["fold", "source", "target"] TODO: Why is this?
    )
)
def create_model_input_nodes(
    graph: KnowledgeGraph,
    splits: pd.DataFrame,
    generator: SingleLabelPairGenerator,
    splitter: BaseCrossValidator = None,
) -> pd.DataFrame:
    """Function to enrich the splits with drug-disease pairs.

    The generator is used to enrich the dataset with unknown drug-disease
    pairs. If a `IterativeDrugDiseasePair` generator is provided, the splits
    dataset is replicated.

    Args:
        graph: Knowledge graph.
        splits: Data splits.
        generator: SingleLabelPairGenerator instance.
        splitter: The splitter used to create the splits. Required to ensure correct generator is used.

    Returns:
        Data with enriched splits.
    """
    if splits.empty:
        raise ValueError("Splits dataframe must be non-empty")

    all_generated = []

    # Enrich splits for all folds
    num_folds = splits["fold"].max() + 1
    for fold in range(num_folds):
        splits_fold = splits[splits["fold"] == fold]
        generated = generator.generate(graph, splits_fold)
        generated["split"] = "TRAIN"
        generated["fold"] = fold
        all_generated.append(generated)

    return pd.concat([splits, *all_generated], axis="index", ignore_index=True)


@inject_object()
def fit_transformers(
    data: pd.DataFrame,
    transformers: dict[str, dict[str, Union[_BaseImputer, list[str]]]],
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
    transformers: dict[str, dict[str, Union[_BaseImputer, list[str]]]],
) -> pd.DataFrame:
    """Function apply fitted transformers to the data.

    Args:
        data: Data to transform.
        transformers: Dictionary of transformers.

    Returns:
        Transformed data.
    """
    for transformer in transformers.values():
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
    features: list[str],
    target_col_name: str,
) -> tuple[dict,]:
    """Function to apply hyperparameter tuning.

    Args:
        data: Data to tune on.
        tuner: Tuner object.
        features: list of features, may be regex specified.
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
                OBJECT_KW: f"{type(estimator).__module__}.{type(estimator).__name__}",
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
    features: list[str],
    target_col_name: str,
) -> dict:
    """Function to train model on the given data.

    Args:
        data: Data to train on.
        estimator: sklearn compatible estimator.
        features: list of features, may be regex specified.
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
        agg_func: function to  aggregate ensemble models' treat score
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
    features: list[str],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> pd.DataFrame:
    """Function to run model class predictions on input data.

    Args:
        data: Data to predict on.
        model: Model making the predictions.
        features: list of features, may be regex specified.
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
    metrics: list[callable],
    target_col_name: str,
    prediction_suffix: str = "_pred",
) -> dict:
    """Function to evaluate model performance on the training data and ground truth test data.

    NOTE: This function only provides a partial indication of model performance,
    primarily for checking that a model has been successfully trained.

    FUTURE: Move to evaluation pipeline.

    Args:
        data: Data to evaluate.
        metrics: list of callable metrics.
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
