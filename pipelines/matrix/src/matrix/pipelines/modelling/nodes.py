import itertools
import json
import logging
from typing import Any, Callable, Iterable, Sequence, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.types as T
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.sql import functions as f
from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import BaseCrossValidator

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from matrix.inject import OBJECT_KW, inject_object, make_list_regexable, unpack_params
from matrix.pipelines.modelling.transformers import WeightingTransformer

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
        disease_list: disease list from https://github.com/everycure-org/core-entities/
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
        # unique=["fold", "source", "target"] TODO: Why is this? - prevent duplicate pairs within folds, unknown if observed issue?
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
    mask = data["split"].eq("TRAIN")
    target = data.loc[mask, target_col_name] if target_col_name else None

    fitted = {}
    for name, meta in transformers.items():
        feats = meta["features"]
        tr = meta["transformer"].fit(data.loc[mask, feats], target)

        fitted[name] = {"transformer": tr, "features": feats}

    return fitted


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

    if "weight" not in data.columns:
        data["weight"] = 1.0

    return data


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable_kwarg="features")
def tune_parameters(data: pd.DataFrame, tuner: Any, features: list[str], target_col_name: str) -> tuple[dict,]:
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
    weights = data.loc[mask, "weight"].values.ravel() if "weight" in data.columns else None

    # Fit tuner
    tuner.fit(X_train.values, y_train.values, sample_weight=weights)

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
def train_model(data: pd.DataFrame, estimator: BaseEstimator, features: list[str], target_col_name: str) -> dict:
    """Fit the final classifier on one fold (TRAIN split only)."""

    mask = data["split"].eq("TRAIN")
    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]
    weights = data.loc[mask, "weight"].values.ravel() if "weight" in data.columns else None

    logger.info(f"Training model ({type(estimator).__name__}) ...")
    estimator_fit = estimator.fit(X_train.values, y_train.values, sample_weight=weights)
    logger.info("Model training completed.")
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


def _tree_shap_values_single(
    booster: xgb.Booster,
    X: pd.DataFrame,
    feature_names: list[str],
    class_idx: int | None = 1,
) -> np.ndarray:
    """Compute SHAP values for a single booster model.

    Args:
        booster: xgboost.Booster model.
        X: DataFrame containing the input features.
        feature_names: List of feature names.
        class_idx: Index of the class for which to compute SHAP values (default is 1).
    Returns:
        np.ndarray: SHAP values for the input features.
    """
    explainer = shap.TreeExplainer(booster, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        if class_idx is None:
            shap_vals = np.mean(shap_vals, axis=0)
        else:
            shap_vals = shap_vals[class_idx]
    elif shap_vals.ndim == 3:
        if shap_vals.shape[1] == len(feature_names):
            feat_axis, class_axis = 1, 2
        else:
            feat_axis, class_axis = 2, 1
        if class_idx is None:
            shap_vals = shap_vals.mean(axis=class_axis)
        else:
            shap_vals = np.take(shap_vals, class_idx, axis=class_axis)

        if feat_axis != 1:
            shap_vals = np.moveaxis(shap_vals, feat_axis, 1)

    if shap_vals.shape[1] == len(feature_names) + 1:
        shap_vals = shap_vals[:, :-1]

    return shap_vals


def _aggregate(stack: np.ndarray, agg_func: Callable) -> np.ndarray:
    """Aggregate a stack of SHAP values using the specified aggregation function.
    Args:
        stack: 3‑D array of SHAP values (n_models, n_rows, n_features).
        agg_func: Function to aggregate SHAP values across models.
    Returns:
        np.ndarray: Aggregated SHAP values (n_rows, n_features).
    """
    try:
        out = agg_func(stack, axis=0)
    except TypeError:
        out = np.apply_along_axis(agg_func, 0, stack)

    if out.ndim != 2:
        raise ValueError(
            f"Aggregation produced shape {out.shape}; "
            "expected 2‑D (n_rows, n_features). "
            "Make sure `agg_func` returns a scalar for a 1‑D input."
        )
    return out


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

    preds = model.predict(data[features].values)

    scored_df = data.copy()
    scored_df[target_col_name + prediction_suffix] = preds

    shap_arrays = [_tree_shap_values_single(bst, data[features], features, class_idx=1) for bst in model.boosters]

    shap_stack = np.stack(shap_arrays)
    shap_agg = _aggregate(shap_stack, model._agg_func)
    shap_df = pd.DataFrame(
        shap_agg,
        index=data.index,
        columns=[f"{c}_shap" for c in features],
    )

    if callable(model._agg_func) and model._agg_func in (np.mean, np.average, np.sum):
        base = np.mean([shap.TreeExplainer(b).expected_value for b in model.boosters])
        recon = base + shap_df.sum(axis=1).values

    if mlflow.active_run():
        explainer = shap.TreeExplainer(model.boosters[0], feature_perturbation="interventional")
        mlflow.shap.log_explainer(explainer, artifact_path="shap")

        fig = plt.figure()
        shap.summary_plot(shap_df.values, data[features], show=False, max_display=20)
        mlflow.log_figure(fig, "shap/summary_beeswarm.png")
        plt.close(fig)

    return scored_df, shap_df


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


# def plot_gt_weights(
#     fitted_tr: WeightingTransformer,
#     train_df: ps.DataFrame,
# ) -> plt.gcf():
#     """
#     Build a diagnostic plot for a *fitted* WeightingTransformer.

#     Parameters
#     ----------
#     fitted_tr : WeightingTransformer
#         Must already be fitted (weight_map_ exists).
#     train_df : pd.DataFrame
#         The TRAIN rows used in `.fit()`. Must contain `fitted_tr.head_col`.

#     Returns
#     -------
#     matplotlib.figure.Figure
#     """
#     head_col = fitted_tr.head_col

#     degrees = train_df.groupby(head_col).size()
#     raw_cnt = train_df[head_col].map(degrees).to_numpy()

#     weights = train_df[head_col].map(fitted_tr.weight_map_).fillna(fitted_tr.default_weight_).to_numpy()
#     w_cnt = raw_cnt * weights

#     bins = max(10, int(np.sqrt(raw_cnt.size)))
#     strategy = fitted_tr.strategy

#     fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=110)
#     ax[0].scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
#     ax[0].set(
#         xlabel="raw degree",
#         ylabel="weighted degree",
#         title=f"{strategy} – mapping",
#     )

#     for vec, col, lab in [
#         (raw_cnt, "tab:blue", "raw"),
#         (w_cnt, "tab:orange", "weighted"),
#     ]:
#         sns.histplot(
#             vec,
#             bins=bins,
#             ax=ax[1],
#             color=col,
#             edgecolor="black",
#             alpha=0.30,
#             stat="count",
#             label=f"{lab} (hist)",
#         )
#         sns.kdeplot(
#             vec,
#             ax=ax[1],
#             color=col,
#             bw_adjust=1.2,
#             linewidth=2,
#             fill=False,
#             label=f"{lab} KDE",
#         )

#     ax[1].set(
#         xlabel="degree",
#         ylabel="entity count",
#         title=f"{strategy} – distribution",
#     )
#     ax[1].legend()

#     def _stats(v):
#         m = v.mean()
#         sd = v.std(ddof=1)
#         return [m, np.median(v), sd, sd / m]

#     rows = np.vstack([_stats(raw_cnt), _stats(w_cnt)])
#     tbl = plt.table(
#         cellText=np.round(rows, 3),
#         colLabels=["mean", "median", "std", "RSD"],
#         rowLabels=["raw", "weighted"],
#         bbox=[0.65, -0.24, 0.33, 0.16],
#     )
#     tbl.auto_set_font_size(True)
#     plt.tight_layout()

#     return plt.gcf()


def plot_gt_weights(*inputs) -> Figure:
    """
    Positional inputs:
        0 … N‑1   : dicts produced by `fit_transformers`
        N         : `modelling.model_input.splits@pandas`
    """
    all_splits = inputs[-1]  # the original DataFrame (with 'source')
    if not isinstance(all_splits, pd.DataFrame):
        all_splits = all_splits.to_pandas()

    tr_dicts: Sequence[dict] = inputs[:-1]
    n = len(tr_dicts)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 6 * n), dpi=110, squeeze=False)

    for i, tr_dict in enumerate(tr_dicts):
        tr: WeightingTransformer = tr_dict["weighting"]["transformer"]

        # TRAIN rows that were used to fit this transformer
        df = all_splits.loc[(all_splits["fold"] == i) & (all_splits["split"] == "TRAIN")]

        head = tr.head_col  # 'source'
        raw_cnt = df.groupby(head).size()
        raw_cnt = df[head].map(raw_cnt).to_numpy()

        weights = df[head].map(tr.weight_map_).fillna(tr.default_weight_).to_numpy()
        w_cnt = raw_cnt * weights
        bins = max(10, int(np.sqrt(raw_cnt.size)))

        ax0, ax1 = axes[i]

        # left panel – mapping
        ax0.scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
        ax0.set(
            xlabel="raw degree",
            ylabel="weighted degree",
            title=f"{tr.strategy} – mapping (fold {i})",
        )

        # right panel – distribution
        for vec, col, lab in [
            (raw_cnt, "tab:blue", "raw"),
            (w_cnt, "tab:orange", "weighted"),
        ]:
            sns.histplot(
                vec,
                bins=bins,
                ax=ax1,
                color=col,
                edgecolor="black",
                alpha=0.30,
                stat="count",
                label=f"{lab} (hist)",
            )
            sns.kdeplot(
                vec,
                ax=ax1,
                color=col,
                bw_adjust=1.2,
                linewidth=2,
                fill=False,
                label=f"{lab} KDE",
            )

        ax1.set(
            xlabel="degree",
            ylabel="entity count",
            title=f"{tr.strategy} – distribution (fold {i})",
        )
        ax1.legend()

    plt.tight_layout()
    return fig
