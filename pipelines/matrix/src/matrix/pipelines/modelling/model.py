"""Module to represent models."""
from typing import List, Callable, Optional
from sklearn.base import BaseEstimator

import numpy as np


class ModelWrapper:
    """Class to represent models.

    TODO: Conventions for different model parts
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
        """Model predict method.

        Args:
            X: input features
        """
        return self._estimators[0].predict(X)  # TODO: Update to aggregate results

    def predict_proba(self, X):
        """Model predict_proba method.

        Args:
            X: input features
        """
        raise NotImplementedError("Predict method not implemented yet")
