from typing import List

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection._split import _BaseKFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space.space import Dimension
from skopt.utils import use_named_args

from .utils import to_estimator_device


class NopTuner(BaseEstimator, MetaEstimatorMixin):
    """No-operation hyperparam tuner.

    Tuner that yields configuration of the input estimator. To use
    in cases when no hyperparameter tuning is required. The NopTuner directly
    yields the sklearn compatible estimator as provided during initialization.
    """

    def __init__(self, estimator: BaseEstimator, **kwargs):
        """Initialize the tuner.

        Args:
            estimator: sklearn compatible Estimator to tune.
            **kwargs: Convenience argument that allows to pass unused parameters
                to the estimator.
        """
        self.estimator = estimator
        super().__init__()

    def fit(self, X, y=None, **params):
        """Function to tune the hyperparameters of the estimator.

        Args:
            X: Feature values
            y: Target values
            **params: Additional parameters to pass to the tuner.

        Returns:
            Fitted estimator.
        """
        self.best_params_ = self.estimator.get_params()

        return self.estimator


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
        self.estimator = estimator
        self.dimensions = dimensions
        self.scoring = scoring
        self.splitter = splitter
        self.n_calls = n_calls
        super().__init__()

    def fit(self, X, y=None, **params):
        """Function to tune the hyperparameters of the estimator.

        Args:
            X: Feature values
            y: Target values
            **params: Additional parameters to pass to the tuner.

        Returns:
            Fitted estimator.
        """

        @use_named_args(self.dimensions)
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
            self.estimator.set_params(**params)

            scores = []
            for train, test in self.splitter.split(X, y):
                X_train_split = to_estimator_device(X[train], self.estimator)
                self.estimator.fit(X_train_split, y[train])
                X_test_split = to_estimator_device(X[test], self.estimator)
                y_pred = self.estimator.predict(X_test_split)
                scores.append(self.scoring(y_pred, y[test]))

            return 1.0 - np.average(scores)

        result = gp_minimize(evaluate_model, self.dimensions, n_calls=self.n_calls, n_jobs=-1)

        self.convergence_plot = plot_convergence(result).figure

        self.best_params_ = {param.name: val for param, val in zip(self.dimensions, result.x)}

        return self.estimator.set_params(**params)
