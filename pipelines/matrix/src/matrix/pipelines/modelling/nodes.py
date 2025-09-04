import itertools
import json
import logging
from inspect import signature
from typing import Any, Callable, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.types as T
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.errors import AnalysisException
from pyspark.sql import functions as f
from sklearn.base import BaseEstimator
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import BaseCrossValidator

from matrix.datasets.graph import KnowledgeGraph
from matrix.datasets.pair_generator import SingleLabelPairGenerator
from matrix.inject import OBJECT_KW, inject_object, make_list_regexable, unpack_params

from .model import ModelWrapper
from .model_selection import DiseaseAreaSplit

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


def _filter_ground_truth(edges_gt: ps.DataFrame, training_data_sources: list[str]) -> ps.DataFrame:
    """
    Filter ground truth to only include pairs from training data sources and drop potential duplicates
    (due to multiple sources for the same pair).

    NOTE: Adding try/except mechanism to avoid breaking main pipeline
    when using older releases of the pipeline

    Args:
        edges_gt: DataFrame with ground truth pairs
        training_data_sources: list of training data sources

    Returns:
        DataFrame with ground truth pairs filtered to only include pairs from training data sources
        and dropped potential duplicates (due to multiple sources for the same pair).
    """
    try:
        return edges_gt.filter(f.col("upstream_data_source").isin(training_data_sources)).dropDuplicates(
            ["subject", "object"]
        )
    except AnalysisException as e:
        logger.error(f"Upstream data source column not found in ground truth; using full dataset")
        return edges_gt


def filter_valid_pairs(
    nodes: ps.DataFrame,
    edges_gt: ps.DataFrame,
    # training_data_sources: list[str],
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

    # Select Ground truth from training data sources and drop potential duplicates
    # (due to multiple sources for the same pair)
    # edges_gt = _filter_ground_truth(edges_gt, training_data_sources)

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
        if name == "weighting":
            tr = meta["transformer"].fit(data.loc[mask, feats + ["y"]], target)
        else:
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
    for name, transformer in transformers.items():
        # Apply transformer
        features = transformer["features"]
        features_selected = data[features]

        if name == "weighting":
            y = data["y"] if "y" in data.columns else None
            transformed_array = transformer["transformer"].transform(features_selected, y=y)
        else:
            transformed_array = transformer["transformer"].transform(features_selected)

        transformed = pd.DataFrame(
            transformed_array,
            index=features_selected.index,
            columns=transformer["transformer"].get_feature_names_out(features_selected),
        )

        # Overwrite columns
        data = pd.concat(
            [data.drop(columns=features), transformed],
            axis="columns",
        )

    # if "weight" not in data.columns:
    #     data["weight"] = 1.0

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


def _build_xyw_from_embeddings(df: pd.DataFrame, head_col="source", tail_col="target"):
    # Build X by concatenating source/target embedding arrays.
    # Assumes columns 'source_embedding' and 'target_embedding' exist with list/array values.
    X = np.array(
        [
            np.concatenate([np.asarray(se, dtype=float), np.asarray(te, dtype=float)])
            for se, te in zip(df["source_embedding"], df["target_embedding"])
        ]
    )
    y = df["y"].to_numpy()
    return X, y


def _global_topn_metrics(
    df_scores: pd.DataFrame,
    n_list: list[int],
    y_col: str = "y",
    score_col: str = "score",
    source_col: str = "source",
    target_col: str = "target",
) -> dict[str, list[float]]:
    """Compute global top-n metrics: recall@n, entropy@n, coverage@n, effective entities fraction@n.

    Args:
        df_scores: pandas DataFrame with at least columns [y_col, score_col, source_col, target_col].
        n_list: list of n values (not necessarily sorted); cumulative metrics are computed for each n.
        y_col: binary label column (1 = positive).
        score_col: score used for ranking (higher is better).
        source_col: column name for source entities (e.g., drug).
        target_col: column name for target entities (e.g., disease).

    Returns:
        A dictionary of lists (aligned to n_list, sorted ascending):
          - recall_at_n
          - source_entropy_at_n
          - target_entropy_at_n
          - source_coverage_at_n
          - target_coverage_at_n
          - source_effective_frac_at_n
          - target_effective_frac_at_n
    """
    if df_scores.empty:
        n_out = [int(n) for n in sorted({n for n in n_list if int(n) > 0})]
        zeros = [0.0] * len(n_out)
        return {
            "n": n_out,
            "recall_at_n": zeros,
            "source_entropy_at_n": zeros,
            "target_entropy_at_n": zeros,
            "source_coverage_at_n": zeros,
            "target_coverage_at_n": zeros,
            "source_effective_frac_at_n": zeros,
            "target_effective_frac_at_n": zeros,
        }

    n_vals = [int(n) for n in sorted({n for n in n_list if int(n) > 0})]
    if not n_vals:
        return {
            "n": [],
            "recall_at_n": [],
            "source_entropy_at_n": [],
            "target_entropy_at_n": [],
            "source_coverage_at_n": [],
            "target_coverage_at_n": [],
            "source_effective_frac_at_n": [],
            "target_effective_frac_at_n": [],
        }

    df = df_scores.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    N_pos = int((df[y_col] == 1).sum())
    if N_pos == 0:
        recall_at_n = [0.0 for _ in n_vals]
    else:
        pos_ranks = (df.index[df[y_col] == 1].to_series(index=None) + 1).to_numpy()
        recall_at_n = [(pos_ranks <= n).sum() / N_pos for n in n_vals]

    src_uniques = pd.Index(df[source_col].unique())
    tgt_uniques = pd.Index(df[target_col].unique())
    M_src = len(src_uniques)
    M_tgt = len(tgt_uniques)

    src_index = {ent: i for i, ent in enumerate(src_uniques)}
    tgt_index = {ent: i for i, ent in enumerate(tgt_uniques)}

    src_counts = np.zeros(M_src, dtype=np.int64)
    tgt_counts = np.zeros(M_tgt, dtype=np.int64)

    def _H_nat_and_norm(counts: np.ndarray, M: int) -> tuple[float, float]:
        if M <= 1:
            return 0.0, 0.0
        total = counts.sum()
        if total <= 0:
            return 0.0, 0.0
        p = counts / total
        mask = p > 0
        H_nat = float(-(p[mask] * np.log(p[mask])).sum())
        H_norm = H_nat / float(np.log(M)) if M > 1 else 0.0
        return H_nat, H_norm

    source_entropy_at_n: list[float] = []
    target_entropy_at_n: list[float] = []
    source_coverage_at_n: list[float] = []
    target_coverage_at_n: list[float] = []
    source_effective_frac_at_n: list[float] = []
    target_effective_frac_at_n: list[float] = []

    prev = 0
    N_rows = len(df)
    for n in n_vals:
        end = min(n, N_rows)
        if end > prev:
            seg = df.iloc[prev:end, [df.columns.get_loc(source_col), df.columns.get_loc(target_col)]]
            src_vals, src_cnts = np.unique(seg[source_col].to_numpy(), return_counts=True)
            for ent, c in zip(src_vals, src_cnts):
                idx = src_index.get(ent)
                if idx is not None:
                    src_counts[idx] += int(c)
            tgt_vals, tgt_cnts = np.unique(seg[target_col].to_numpy(), return_counts=True)
            for ent, c in zip(tgt_vals, tgt_cnts):
                idx = tgt_index.get(ent)
                if idx is not None:
                    tgt_counts[idx] += int(c)
            prev = end

        Hs_nat, Hs_norm = _H_nat_and_norm(src_counts, M_src)
        Ht_nat, Ht_norm = _H_nat_and_norm(tgt_counts, M_tgt)
        source_entropy_at_n.append(Hs_norm)
        target_entropy_at_n.append(Ht_norm)

        src_cov = (src_counts > 0).sum() / M_src if M_src > 0 else 0.0
        tgt_cov = (tgt_counts > 0).sum() / M_tgt if M_tgt > 0 else 0.0
        source_coverage_at_n.append(float(src_cov))
        target_coverage_at_n.append(float(tgt_cov))

        src_eff = float(np.exp(Hs_nat)) / M_src if M_src > 0 else 0.0
        tgt_eff = float(np.exp(Ht_nat)) / M_tgt if M_tgt > 0 else 0.0
        source_effective_frac_at_n.append(src_eff)
        target_effective_frac_at_n.append(tgt_eff)

    return {
        "n": n_vals,
        "recall_at_n": recall_at_n,
        "source_entropy_at_n": source_entropy_at_n,
        "target_entropy_at_n": target_entropy_at_n,
        "source_coverage_at_n": source_coverage_at_n,
        "target_coverage_at_n": target_coverage_at_n,
        "source_effective_frac_at_n": source_effective_frac_at_n,
        "target_effective_frac_at_n": target_effective_frac_at_n,
    }


@inject_object()
def tune_weighting_optuna(
    data: pd.DataFrame,
    base_transformers: dict,
    tuning: dict,
    estimator_params: dict,
    features: list[str],
    target_col_name: str,
    seed: int,
    storage_url: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = True,
) -> dict:
    if not tuning or not tuning.get("enabled", False):
        return base_transformers

    try:
        import optuna
    except Exception as e:
        logger.warning(f"Optuna not available ({e}); using base transformers without tuning.")
        return base_transformers

    df = data.copy()
    mask_train = df["split"].eq("TRAIN")
    df_train = df.loc[mask_train]
    df_test = df.loc[~mask_train]
    if df_test.empty or df_train.empty:
        logger.warning("Empty TRAIN or TEST slice for weighting tuning; skipping.")
        return base_transformers

    k = int(tuning.get("k", 100))
    obj = tuning.get("objective", {})
    w_recall = float(obj.get("w_recall", 1.0))
    w_entropy = float(obj.get("w_entropy", 0.0))
    w_bounds = float(obj.get("w_bounds", 0.0))
    bounds_target = float(obj.get("bounds_target", 0.2))
    space = tuning.get("search_space", {}) or {}
    w_head_bias = float(obj.get("w_head_bias", 0.0))
    w_corr = float(obj.get("w_corr", 0.0))
    from xgboost import XGBClassifier

    est_kwargs = {}
    if isinstance(estimator_params, BaseEstimator):
        try:
            est_kwargs = estimator_params.get_params()
        except Exception:
            est_kwargs = {}
    elif isinstance(estimator_params, dict):
        est_kwargs = dict(estimator_params)
    else:
        est_kwargs = {}

    try:
        if isinstance(estimator_params, dict) and "random_state" in estimator_params:
            seed = int(estimator_params["random_state"])
        elif "random_state" in tuning:
            seed = int(tuning["random_state"])
    except Exception:
        seed = 42

    est_kwargs.pop("_object", None)
    try:
        valid_params = set(signature(XGBClassifier.__init__).parameters.keys())
        est_kwargs = {k: v for k, v in est_kwargs.items() if k in valid_params}
    except Exception:
        pass

    est_kwargs.setdefault("n_estimators", 300)
    est_kwargs.setdefault("learning_rate", 0.1)
    est_kwargs.setdefault("max_depth", 6)
    est_kwargs.setdefault("random_state", seed)

    est = XGBClassifier(**est_kwargs)

    X_tr, y_tr = _build_xyw_from_embeddings(df_train)
    heads_tr = df_train["source"].to_numpy()
    tails_tr = df_train["target"].to_numpy()

    from sklearn.model_selection import StratifiedShuffleSplit

    val_size = float(tuning.get("val_size", 0.2))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx_tr_in, idx_val_in = next(sss.split(X_tr, y_tr))

    X_tr_in, y_tr_in = X_tr[idx_tr_in], y_tr[idx_tr_in]
    X_val_in, y_val_in = X_tr[idx_val_in], y_tr[idx_val_in]
    heads_tr_in = heads_tr[idx_tr_in]
    heads_val_in = heads_tr[idx_val_in]
    tails_tr_in = tails_tr[idx_tr_in]
    tails_val_in = tails_tr[idx_val_in]

    base_w_conf = (
        base_transformers["weighting"]["transformer"].get_params()
        if hasattr(base_transformers["weighting"]["transformer"], "get_params")
        else base_transformers["weighting"]["transformer"]
    )
    base_conf = (
        dict(base_w_conf) if isinstance(base_w_conf, dict) else dict(base_transformers["weighting"]["transformer"])
    )

    try:
        fold_id = int(pd.unique(df["fold"])[0]) if "fold" in df.columns else None
    except Exception:
        fold_id = None
    default_study_name = f"weighting_fold_{fold_id}" if fold_id is not None else "weighting"

    import json

    def _flatten_dict(d: dict | None, prefix: str = "") -> dict[str, object]:
        out = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_dict(v, key))
            else:
                out[key] = v
        return out

    def _json_safe(v):
        import numpy as np
        import pandas as pd

        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
        if isinstance(v, (pd.Series, pd.Index)):
            return v.to_list()
        if isinstance(v, (float, int, str, bool)) or v is None:
            return v
        try:
            json.dumps(v)
            return v
        except Exception:
            try:
                return str(v)
            except Exception:
                return None

    def suggest_params(trial: "optuna.Trial") -> dict:
        conf = dict(base_conf)
        for key, bounds in space.items():
            low, high = float(bounds[0]), float(bounds[1])
            conf[key] = trial.suggest_float(key, low, high)
        return conf

    def objective(trial: "optuna.Trial") -> float:
        import numpy as np

        conf = suggest_params(trial)
        from matrix.pipelines.modelling.transformers import WeightingTransformer

        wt = WeightingTransformer(**conf)

        df_fit_in = df_train.iloc[idx_tr_in][["source", "target", "y"]].copy()
        wt.fit(df_fit_in)

        w_tr_in = wt.transform(df_train.iloc[idx_tr_in][["source", "target"]], y=y_tr_in).ravel()

        fit_sig = signature(est.fit).parameters
        fit_kwargs = {}

        if "sample_weight" in fit_sig:
            fit_kwargs["sample_weight"] = w_tr_in
        if "eval_set" in fit_sig:
            fit_kwargs["eval_set"] = [(X_val_in, y_val_in)]
        if "eval_metric" in fit_sig:
            fit_kwargs["eval_metric"] = tuning.get("eval_metric", "auc")

        es_rounds = int(tuning.get("early_stopping_rounds", 50))
        if "early_stopping_rounds" in fit_sig:
            fit_kwargs["early_stopping_rounds"] = es_rounds
            est.fit(X_tr_in, y_tr_in, **fit_kwargs)
        elif "callbacks" in fit_sig:
            try:
                from xgboost.callback import EarlyStopping

                fit_kwargs["callbacks"] = [EarlyStopping(rounds=es_rounds, save_best=True)]
            except Exception:
                pass
            est.fit(X_tr_in, y_tr_in, **fit_kwargs)
        else:
            est.fit(X_tr_in, y_tr_in, **fit_kwargs)

        y_score = est.predict_proba(X_val_in)[:, 1]
        df_scores = pd.DataFrame({"y": y_val_in, "score": y_score, "source": heads_val_in, "target": tails_val_in})

        metrics = _global_topn_metrics(
            df_scores,
            n_list=[k],
            y_col="y",
            score_col="score",
            source_col="source",
            target_col="target",
        )
        recall_k = float(metrics["recall_at_n"][0])
        ent_src_k = float(metrics["source_entropy_at_n"][0])
        ent_tgt_k = float(metrics["target_entropy_at_n"][0])
        cov_src_k = float(metrics["source_coverage_at_n"][0])
        cov_tgt_k = float(metrics["target_coverage_at_n"][0])
        eff_src_k = float(metrics["source_effective_frac_at_n"][0])
        eff_tgt_k = float(metrics["target_effective_frac_at_n"][0])

        entropy_mode = str(tuning.get("objective", {}).get("entropy_mode", "source")).lower()
        if entropy_mode in ("avg", "mean"):
            ent_k = 0.5 * (ent_src_k + ent_tgt_k)
            cov_k = 0.5 * (cov_src_k + cov_tgt_k)
            eff_k = 0.5 * (eff_src_k + eff_tgt_k)
        elif entropy_mode == "target":
            ent_k, cov_k, eff_k = ent_tgt_k, cov_tgt_k, eff_tgt_k
        else:
            entropy_mode = "source"
            ent_k, cov_k, eff_k = ent_src_k, cov_src_k, eff_src_k

        df_bias = pd.DataFrame({"source": heads_tr_in, "w": w_tr_in, "one": 1.0})
        deg_raw = df_bias.groupby("source")["one"].sum().astype(float)
        deg_w = df_bias.groupby("source")["w"].sum()

        if len(deg_w) > 1:
            p_w = deg_w / deg_w.sum()
            H_w = float(-(p_w * np.log(p_w + 1e-12)).sum())
            H_max = float(np.log(len(p_w)))
            ent_train_norm = H_w / H_max
        else:
            ent_train_norm = 0.0
        head_bias_penalty = 1.0 - ent_train_norm

        if len(deg_raw) > 1:
            aligned = deg_raw.align(deg_w, join="inner")
            rho = float(aligned[0].corr(aligned[1], method="spearman"))
            if np.isnan(rho):
                rho = 0.0
        else:
            rho = 0.0
        corr_penalty = abs(rho)

        stats_neg = getattr(wt, "fit_stats_", {}).get("neg", {})
        pct_at = float(stats_neg.get("pct_a_at_bounds", 0.0))
        bounds_penalty = max(0.0, pct_at - bounds_target)

        score = (
            (w_recall * recall_k)
            + (w_entropy * ent_k)
            - (w_bounds * bounds_penalty)
            - (w_head_bias * head_bias_penalty)
            - (w_corr * corr_penalty)
        )

        trial.set_user_attr("k", int(k))
        trial.set_user_attr("entropy_mode", entropy_mode)
        trial.set_user_attr("recall@N", recall_k)
        trial.set_user_attr("entropy@N", ent_k)

        trial.set_user_attr("source_entropy@N", ent_src_k)
        trial.set_user_attr("target_entropy@N", ent_tgt_k)

        trial.set_user_attr("source_coverage@N", cov_src_k)
        trial.set_user_attr("target_coverage@N", cov_tgt_k)
        trial.set_user_attr("coverage@N", cov_k)

        trial.set_user_attr("source_effective_frac@N", eff_src_k)
        trial.set_user_attr("target_effective_frac@N", eff_tgt_k)
        trial.set_user_attr("effective_entities_frac@N", eff_k)
        trial.set_user_attr("effective_heads_frac@N", eff_k)

        trial.set_user_attr("pct_a_at_bounds_neg", pct_at)

        trial.set_user_attr("head_bias_penalty", head_bias_penalty)
        trial.set_user_attr("corr_penalty", corr_penalty)

        wt_params = wt.get_params(deep=False) if hasattr(wt, "get_params") else conf
        for p_key, p_val in wt_params.items():
            trial.set_user_attr(f"wparam.{p_key}", _json_safe(p_val))

        fit_stats = getattr(wt, "fit_stats_", {}) or {}
        for f_key, f_val in _flatten_dict(fit_stats, prefix="wfit").items():
            trial.set_user_attr(f_key, _json_safe(f_val))

        try:
            fold_id = int(pd.unique(df["fold"])[0]) if "fold" in df.columns else None
        except Exception:
            fold_id = None
        if fold_id is not None:
            trial.set_user_attr("fold", int(fold_id))

        return score

    def _sqlite_local_path_from_url(url: str) -> str:
        if url.startswith("sqlite:////"):
            return url[len("sqlite:") :].lstrip("/")
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///") :]
        return url

    import os
    import tempfile
    from pathlib import Path

    def _make_sqlite_url(local_path: str) -> str:
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_absolute():
            return f"sqlite:////{p.as_posix().lstrip('/')}"
        return f"sqlite:///{p.as_posix()}"

    storage_for_optuna = None
    post_sync = None

    if storage_url:
        s = str(storage_url)
        if s.startswith("gs://"):
            try:
                import sqlite3

                from google.cloud import storage as gcs

                _, _, rest = s.partition("gs://")
                bucket_name, _, blob_name = rest.partition("/")

                cache_root = Path(os.environ.get("OPTUNA_GCS_CACHE", Path(tempfile.gettempdir()) / "optuna_gcs"))
                local_db_path = cache_root / bucket_name / blob_name
                local_db_path.parent.mkdir(parents=True, exist_ok=True)

                client = gcs.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                if blob.exists(client):
                    blob.download_to_filename(local_db_path.as_posix())
                else:
                    sqlite3.connect(local_db_path.as_posix()).close()
                    blob.upload_from_filename(local_db_path.as_posix())

                storage_for_optuna = _make_sqlite_url(local_db_path.as_posix())

                def _post_sync():
                    try:
                        blob.upload_from_filename(local_db_path.as_posix())
                        logger.info(f"Uploaded Optuna DB to {storage_url}")
                    except Exception as e:
                        logger.warning(f"GCS sync (post) failed; DB remains only local at {local_db_path}: {e}")

                post_sync = _post_sync

            except Exception as e:
                logger.warning(f"GCS sync disabled ({e}); falling back to local SQLite only.")
                local_db_path = Path(tempfile.gettempdir()) / "optuna_gcs_fallback" / "study.db"
                storage_for_optuna = _make_sqlite_url(local_db_path.as_posix())

        elif s.startswith("sqlite:"):
            storage_for_optuna = s
        else:
            storage_for_optuna = _make_sqlite_url(s)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage_for_optuna,
        study_name=(study_name or default_study_name),
        load_if_exists=load_if_exists,
    )
    study.optimize(objective, n_trials=int(tuning.get("n_trials", 30)))

    if post_sync:
        post_sync()

    best_conf = suggest_params(study.best_trial)

    tuned = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_transformers.items()}
    tr = tuned["weighting"]["transformer"]
    if hasattr(tr, "get_params"):
        try:
            tr.set_params(**best_conf)
        except Exception as e:
            logger.warning(f"Failed to set_params on weighting transformer: {e}")
        tuned["weighting"]["transformer"] = tr
    else:
        tr_conf = dict(tr) if isinstance(tr, dict) else {}
        tr_conf.update(best_conf)
        tuned["weighting"]["transformer"] = tr_conf

    logger.info(
        f"Weighting Optuna best: value={study.best_value:.4f}, "
        f"recall@{k}={study.best_trial.user_attrs.get('recall@N'):.4f}, "
        f"entropy_mode={study.best_trial.user_attrs.get('entropy_mode')}, "
        f"entropy@{k}={study.best_trial.user_attrs.get('entropy@N'):.4f}, "
        f"pct_a_at_bounds_neg={study.best_trial.user_attrs.get('pct_a_at_bounds_neg'):.3f}"
    )
    return tuned


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

    # Compute unweighted and (if supported) weighted metrics
    for name, func in metrics.items():
        accepts_weight = "sample_weight" in signature(func).parameters
        for split in ["TEST", "TRAIN"]:
            split_index = data["split"].eq(split)
            y_true = data.loc[split_index, target_col_name]
            y_pred = data.loc[split_index, target_col_name + prediction_suffix]

            # Unweighted
            report[f"{split.lower()}_{name}"] = func(y_true, y_pred)

            # Weighted (if metric supports it and weights exist)
            if accepts_weight and "weight" in data.columns:
                w = data.loc[split_index, "weight"]
                report[f"{split.lower()}_{name}_weighted"] = func(y_true, y_pred, sample_weight=w)

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


# def plot_gt_weights(*inputs) -> Figure:
#     """
#     Positional inputs:
#         0 … N‑1   : dicts produced by `fit_transformers`
#         N         : `modelling.model_input.splits@pandas`
#     """
#     all_splits = inputs[-1]  # the original DataFrame (with 'source')
#     if not isinstance(all_splits, pd.DataFrame):
#         all_splits = all_splits.to_pandas()

#     tr_dicts: Sequence[dict] = inputs[:-1]
#     n = len(tr_dicts)

#     fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 6 * n), dpi=110, squeeze=False)

#     for i, tr_dict in enumerate(tr_dicts):
#         tr: WeightingTransformer = tr_dict["weighting"]["transformer"]

#         # TRAIN rows that were used to fit this transformer
#         df = all_splits.loc[(all_splits["fold"] == i) & (all_splits["split"] == "TRAIN")]

#         head = tr.head_col  # 'source'
#         raw_cnt = df.groupby(head).size()
#         raw_cnt = df[head].map(raw_cnt).to_numpy()

#         weights = df[head].map(tr.weight_map_).fillna(tr.default_weight_).to_numpy()
#         w_cnt = raw_cnt * weights
#         bins = max(10, int(np.sqrt(raw_cnt.size)))

#         ax0, ax1 = axes[i]

#         # left panel – mapping
#         ax0.scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
#         ax0.set(
#             xlabel="raw degree",
#             ylabel="weighted degree",
#             title=f"{tr.strategy} – mapping (fold {i})",
#         )

#         # right panel – distribution
#         for vec, col, lab in [
#             (raw_cnt, "tab:blue", "raw"),
#             (w_cnt, "tab:orange", "weighted"),
#         ]:
#             sns.histplot(
#                 vec,
#                 bins=bins,
#                 ax=ax1,
#                 color=col,
#                 edgecolor="black",
#                 alpha=0.30,
#                 stat="count",
#                 label=f"{lab} (hist)",
#             )
#             sns.kdeplot(
#                 vec,
#                 ax=ax1,
#                 color=col,
#                 bw_adjust=1.2,
#                 linewidth=2,
#                 fill=False,
#                 label=f"{lab} KDE",
#             )

#         ax1.set(
#             xlabel="degree",
#             ylabel="entity count",
#             title=f"{tr.strategy} – distribution (fold {i})",
#         )
#         ax1.legend()

#     plt.tight_layout()
#     return fig
