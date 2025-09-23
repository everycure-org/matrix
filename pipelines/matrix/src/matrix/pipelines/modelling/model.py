from typing import Callable, List

import numpy as np
import pandas as pd
from matrix_inject.inject import _extract_elements_in_list
from sklearn.base import BaseEstimator

from matrix.pipelines.modelling.nodes import apply_transformers


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


class TransformingModelWrapper:
    """Wrap a trained model with its preprocessing steps.

    This wrapper keeps the fitted transformers and feature selection required
    to run the underlying model. It exposes a sklearn-like ``predict_proba``
    interface so it can be composed inside another ``ModelWrapper``.
    """

    def __init__(
        self,
        transformers: dict,
        model: ModelWrapper,
        features: list[str],
    ) -> None:
        self._transformers = transformers
        self._model = model
        self._features = features

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Run preprocessing and forward the call to the underlying model."""
        transformed = apply_transformers(data.copy(), self._transformers)
        model_features = _extract_elements_in_list(transformed.columns, self._features, True)
        return self._model.predict_proba(transformed[model_features].values)
