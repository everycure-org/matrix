import logging
from typing import Callable, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


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

        Note:
            Uses joblib parallelization to speed up prediction across multiple estimators.
        """
        # NOTE: This function was partially modified using AI assistance.
        all_preds = Parallel(n_jobs=-1)(delayed(estimator.predict_proba)(X) for estimator in self._estimators)
        stacked_preds = np.stack(all_preds)
        return np.apply_along_axis(self._agg_func, 0, stacked_preds)
