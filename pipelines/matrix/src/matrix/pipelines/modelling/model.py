from typing import Callable, List

import numpy as np
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


class ModelWrapper:
    """Class to represent models.

    FUTURE: Add `features` and `transformers` to class, such that we can
    clean up the Kedro viz UI.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        agg_func: Callable,
    ) -> None:
        """Create instance of the model wrapper.

        Args:
            estimators: list of estimators.
            agg_func: function to aggregate ensemble results.
        """
        self._estimators = estimators
        self._agg_func = agg_func
        super().__init__()

    def fit(self, X, y):
        """Model fit method.

        Args:
            X: input features
            y: label
        """
        raise NotImplementedError("ModelWrapper is used to house fitted estimators")

    def predict(self, X):
        """Returns the predicted class.

        Args:
            X: input features
        """
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Method for probability scores of the ModelWrapper.

        Args:
            X: input features

        Returns:
            Aggregated probabilities scores of the individual models.
        """
        all_preds = np.array([estimator.predict_proba(X) for estimator in self._estimators])
        return np.apply_along_axis(self._agg_func, 0, all_preds)


class XGBClassifierWeighted(XGBClassifier):
    """XGBoost classifier that accepts sample weights during training."""

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit the model with optional sample weights.

        Args:
            X: Training data
            y: Target values
            sample_weight: Sample weights
            **kwargs: Additional arguments passed to XGBClassifier.fit()
        """
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        return super().fit(X, y, **kwargs)

    # def fit(self, X, y):
    #     X_reduced = X[:, :-1]
    #     weights = 1/X[:, -1]
    #     return self.model.fit(X_reduced, y, sample_weight = weights)
