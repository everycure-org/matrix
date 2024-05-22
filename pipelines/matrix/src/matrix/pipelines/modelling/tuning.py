from typing import Any, List

import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.metrics import f1_score

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Dimension

from sklearn.model_selection._split import _BaseKFold


def f1_score_df(model: BaseEstimator, X, y):
    """
    Function to calculate the f1 score of the model on the given data.

    Args:
        model: Model to evaluate.
        X: Feature values
        y: Target values
    Returns:
        F1 score of the model.
    """
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

        WARNING: Currently returns the SciPy's OptimizeResult object,
        which is not fully compatible with sklearns' BaseEstimator fit method.

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

            self._estimator.set_params(**params)

            scores = []
            for train, test in self._splitter.split(X, y):
                self._estimator.fit(X[train], y[train])
                scores.append(self._scoring(self._estimator, X[test], y[test]))

            return 1.0 - np.average(scores)

        self.result = gp_minimize(
            evaluate_model, self._dimensions, n_calls=self._n_calls
        )
        self.best_params_ = {
            param.name: self._extract(val)
            for param, val in zip(self._dimensions, self.result.x)
        }

        return self.result

    @staticmethod
    def _extract(val: Any):
        """Helper function to extract items from numpy objects"""
        if isinstance(val, np.generic):
            return val.item()

        return val
