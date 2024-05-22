from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

from sklearn.model_selection._split import _BaseKFold
from sklearn.impute._base import _BaseImputer
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.metrics import f1_score

from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.unpack import unpack_params
from refit.v1.core.make_list_regexable import make_list_regexable

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Dimension

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
    """
    Add features for nodes.

    Args:
        raw_nodes: Raw nodes data.
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
    """
    Create primary pairs dataset.

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
    """
    Function to split data.

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
    """
    Function to enrich the splits with drug-disease pairs.

    Args:
        graph: Knowledge graph.
        splits: Data splits.
        generator: DrugDiseasePairGenerator instance.
    Returns:
        Data with enriched splits.
    """

    # FUTURE: Update the n-unknown
    generated = generator.generate(graph, splits, n_unknown=50)
    generated["split"] = "TRAIN"

    return pd.concat([splits, generated], axis=0, ignore_index=True)


@inject_object()
def apply_transformers(
    data: pd.DataFrame,
    transformers: Dict[str, Dict[str, Union[_BaseImputer, List[str]]]],
    target_col_name: str = None,
) -> pd.DataFrame:
    """
    Function to apply a set of sklearn compatible transformers to the data.

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


def f1_score_df(model: BaseEstimator, X, y):
    y_pred = model.predict(X)
    return f1_score(y_pred, y, average="macro")


class GaussianSearch(BaseEstimator, MetaEstimatorMixin):
    """
    Adaptor class to wrap skopt's gp_minimize into sklearn's BaseEstimator compatible type.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        dimensions: List[Dimension],
        scoring: callable,
        *,
        splitter: _BaseKFold = None,
        n_calls: int = 100,
    ) -> None:
        self._estimator = estimator
        self._dimensions = dimensions
        self._scoring = scoring
        self._splitter = splitter
        self._n_calls = n_calls
        super().__init__()

    def fit(self, X, y=None, **params):
        """
        Function to tune the hyperparameters of the estimator.

        Args:
            X: Feature values
            y: Target values
        Returns:
            Fitted estimator.
        """

        @use_named_args(self._dimensions)
        def evaluate_model(**params):
            """
            Function to evaluate model using the given splitter
            and scoring functions. When the splitter applies kfold splitting,
            the scores are averaged over the folds.
            """
            scores = []
            for _, (train, test) in enumerate(self._splitter.split(X, y)):
                self._estimator.fit(X[train], y[train])
                scores.append(self._scoring(self._estimator, X[test], y[test]))

            return 1.0 - np.average(scores)

        result = gp_minimize(evaluate_model, self._dimensions, n_calls=self._n_calls)
        self.best_params_ = {
            param.name: self._extract(val)
            for param, val in zip(self._dimensions, result.x)
        }

        return self._estimator

    @staticmethod
    def _extract(val: Any):
        """Helper function to extract items from numpy objects"""
        if isinstance(val, np.generic):
            return val.item()

        return val


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def tune_parameters(
    data: pd.DataFrame,
    tuner: Any,
    features: List[str],
    target_col_name: str,
    enable_regex: str = True,
) -> Dict:
    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    # Fit tuner
    tuner.fit(X_train.values, y_train.values)

    return {
        "object": f"{type(tuner._estimator).__module__}.{type(tuner._estimator).__name__}",
        **tuner.best_params_,
    }


@unpack_params()
@inject_object()
@make_list_regexable(source_df="data", make_regexable="features")
def train_model(
    data: pd.DataFrame,
    estimator: BaseGradientBoosting,
    features: List[str],
    target_col_name: str,
    enable_regex: str = True,
) -> Dict:
    """
    Function to train model on the given data.

    Args:
        data: Data to train on.
        estimator: sklearn estimator.
        features: List of features, may be regex specified.
        target_col_name: Target column name.
    """

    mask = data["split"].eq("TRAIN")

    X_train = data.loc[mask, features]
    y_train = data.loc[mask, target_col_name]

    return estimator.fit(X_train.values, y_train.values)
