from typing import List, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


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


class SumSoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A simple classifier that sums features and applies softmax.

    This classifier follows the same interface as other sklearn estimators
    to work with the existing pipeline infrastructure.

    Args:
        random_state: Optional random seed for reproducibility
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit(self, X, y):
        """No training needed for this model."""
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """Sum features and apply softmax activation."""
        # Sum across features
        sums = X.sum(axis=1)

        # Reshape to (n_samples, 1) for binary classification
        scores = np.column_stack([-sums, sums])

        # Apply softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
