from typing import Callable, List

import numpy as np
from sklearn.base import BaseEstimator

from .utils import estimator_uses_cuda, to_estimator_device


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
        cached_inputs = {}
        all_preds = []

        for estimator in self._estimators:
            key = "cuda" if estimator_uses_cuda(estimator) else "cpu"
            if key not in cached_inputs:
                cached_inputs[key] = to_estimator_device(X, estimator) if key == "cuda" else X

            preds = estimator.predict_proba(cached_inputs[key])
            all_preds.append(np.asarray(preds))

        stacked_preds = np.stack(all_preds)
        return np.apply_along_axis(self._agg_func, 0, stacked_preds)
