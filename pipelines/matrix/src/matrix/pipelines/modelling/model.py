import logging
from typing import Callable, List

import numpy as np
from sklearn.base import BaseEstimator

from .utils import to_estimator_device

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
        """
        # Convert input to CUDA once for all estimators
        logger.info("ModelWrapper: Converting input data to estimator device")
        X_cuda = to_estimator_device(X, self._estimators[0])

        logger.info(f"ModelWrapper: Input data converted to estimator device: {X_cuda.device}")
        all_preds = [np.asarray(estimator.predict_proba(X_cuda)) for estimator in self._estimators]
        logger.info(f"ModelWrapper: All predictions collected: {len(all_preds)}")
        stacked_preds = np.stack(all_preds)
        logger.info(f"ModelWrapper: Stacked predictions shape: {stacked_preds.shape}")
        return np.apply_along_axis(self._agg_func, 0, stacked_preds)
