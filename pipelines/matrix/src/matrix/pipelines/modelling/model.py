from typing import List, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax

import numpy as np
import pandas as pd


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


class DotProductSoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that uses dot product between source and target embeddings."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def _prepare_features(self, X):
        """Convert embeddings from DataFrame or array to numpy array."""
        if isinstance(X, pd.DataFrame):
            # Handle PySpark DataFrame conversion to numpy
            if hasattr(X, "toPandas"):
                X = X.toPandas()

            # Convert source embeddings
            if "source_embedding" in X.columns:
                source_embeddings = np.vstack(
                    [np.array(x) if isinstance(x, list) else x for x in X["source_embedding"]]
                )
                target_embeddings = np.vstack(
                    [np.array(x) if isinstance(x, list) else x for x in X["target_embedding"]]
                )
            else:
                # Assume columns are already split into individual features
                source_cols = [col for col in X.columns if col.startswith("source_")]
                target_cols = [col for col in X.columns if col.startswith("target_")]
                source_embeddings = X[source_cols].values
                target_embeddings = X[target_cols].values

            return np.hstack([source_embeddings, target_embeddings])
        elif isinstance(X, np.ndarray) and X.dtype == object:
            source_embeddings = np.vstack([np.array(x) if isinstance(x, list) else x for x in X[:, 0]])
            target_embeddings = np.vstack([np.array(x) if isinstance(x, list) else x for x in X[:, 1]])
            return np.hstack([source_embeddings, target_embeddings])
        return X

    def fit(self, X, y):
        """Fit the model."""
        X = self._prepare_features(X)
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)
        self.embedding_dim_ = X.shape[1] // 2
        return self

    def predict_proba(self, X):
        """Compute probabilities using dot product and softmax."""
        X = self._prepare_features(X)

        # Split features into source and target embeddings
        source_embeddings = X[:, : self.embedding_dim_]
        target_embeddings = X[:, self.embedding_dim_ :]

        # Compute dot product
        dot_products = np.sum(source_embeddings * target_embeddings, axis=1)

        # Reshape to (n_samples, 2) for binary classification
        scores = np.column_stack([-dot_products, dot_products])

        # Apply softmax
        probs = softmax(scores, axis=1)

        return probs

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
