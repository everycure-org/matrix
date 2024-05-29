"""Module with sklearn compatible tuner classes."""
from typing import List

import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Dimension

from sklearn.model_selection._split import _BaseKFold


class GaussianSearch(BaseEstimator, MetaEstimatorMixin):
    """Guassian Process based hyperparameter tuner.

    Adaptor class to wrap skopt's gp_minimize into sklearn's BaseEstimator compatible type.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        dimensions: List[Dimension],
        scoring: callable,
        *,
        splitter: _BaseKFold = None,
        n_calls: int = 100,
    ) -> None:
        """Initialize the tuner.

        Args:
            estimator: sklearn compatible Estimator to tune.
            dimensions: List of dimensions to tune.
            scoring: Scoring function to evaluate the model.
            splitter: Splitter to use for cross-validation.
            n_calls: Number of calls to the objective function.
        """
        self._estimator = estimator
        self._dimensions = dimensions
        self._scoring = scoring
        self._splitter = splitter
        self._n_calls = n_calls
        super().__init__()

    def fit(self, X, y=None, **params):
        """Function to tune the hyperparameters of the estimator.

        WARNING: Currently returns the SciPy's OptimizeResult object,
        which is not fully compatible with sklearns' BaseEstimator fit method.

        Args:
            X: Feature values
            y: Target values
            **params: Additional parameters to pass to the tuner.

        Returns:
            Fitted estimator.
        """

        @use_named_args(self._dimensions)
        def evaluate_model(**params):
            """Function to evaluate model using the given splitter.

            Function evaluates function using the given splitter and scoring
            functions. When the splitter applies kfold splitting, the scores are
            averaged over the folds.

            FUTURE: Expand invocation of scoring function to inject additional
                columns of the input dataframe, e.g., id's of the drug/disease to
                compute more elaborate metrics, e.g., MRR with synthesized negatives,
                without including these columns as features in the trained model.

            Args:
                **params: Parameters to set on the estimator.
            """
            self._estimator.set_params(**params)

            scores = []
            for train, test in self._splitter.split(X, y):
                self._estimator.fit(X[train], y[train])
                y_pred = self._estimator.predict(X[test])
                scores.append(self._scoring(y_pred, y[test]))

            return 1.0 - np.average(scores)

        self.result = gp_minimize(
            evaluate_model, self._dimensions, n_calls=self._n_calls
        )
        self.best_params_ = {
            param.name: val for param, val in zip(self._dimensions, self.result.x)
        }

        return self.result
