"""Module to represent models."""
from typing import List, Callable, Optional
from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    """Class to represent models."""

    def __init__(
        self,
        estimators: List[BaseEstimator],
        agg: Optional[Callable],
    ) -> None:
        """Create instance of the model container.

        Args:
            estimators: list of estimators.
            agg: function to aggregate ensemble results.
        """
        self._estimators = estimators
        self._agg = agg
        super().__init__()

    def predict_proba(self, X):
        """Predict proba."""
        return self._agg([estimator.predict_proba(X) for estimator in self._estimators])
