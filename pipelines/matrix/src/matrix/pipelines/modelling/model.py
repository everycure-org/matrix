"""Module to represent models."""
from typing import List, Callable, Optional
from sklearn.base import BaseEstimator

import numpy as np


class ModelWrapper:
    """Class to represent models."""

    def __init__(
        self,
        estimators: List[BaseEstimator],
    ) -> None:
        """Create instance of the model wrapper.

        Args:
            estimators: list of estimators.
            agg: function to aggregate ensemble results.
        """
        self._estimators = estimators
        super().__init__()

    def predict_proba(self, X):
        """Predict proba.

        FUTURE: Ensure passing in agg. func into wrapper class.
        """
        return np.concatenate(
            [estimator.predict_proba(X) for estimator in self._estimators]
        ).mean(axis=0)
