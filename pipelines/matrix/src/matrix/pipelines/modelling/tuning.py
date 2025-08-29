from inspect import signature
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection._split import _BaseKFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space.space import Dimension
from skopt.utils import use_named_args


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
        self._dimensions = dimensions
        self._scoring = scoring
        self._splitter = splitter
        self._n_calls = n_calls
        super().__init__()

    def fit(self, X, y=None, sample_weight=None, **params):
        """Function to tune the hyperparameters of the estimator.

        Args:
            X: Feature values
            y: Target values
            **params: Additional parameters to pass to the tuner.

        Returns:
            Fitted estimator.
        """

        _scorer_accepts_weight = "sample_weight" in signature(self._scoring).parameters

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
            self.estimator.set_params(**params)

            scores = []
            for train, test in self._splitter.split(X, y):
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]

                weights_train = sample_weight[train] if sample_weight is not None else None
                weights_test = sample_weight[test] if sample_weight is not None else None

                if weights_train is not None:
                    self.estimator.fit(X_train, y_train, sample_weight=weights_train)
                else:
                    self.estimator.fit(X_train, y_train)

                y_pred = self.estimator.predict(X_test)

                if _scorer_accepts_weight and weights_test is not None:
                    score = self._scoring(y_test, y_pred, sample_weight=weights_test)
                else:
                    score = self._scoring(y_test, y_pred)
                # score = self._scoring(y_test, y_pred)

                scores.append(score)

            return 1.0 - np.average(scores)

        result = gp_minimize(evaluate_model, self._dimensions, n_calls=self._n_calls)

        self.convergence_plot = plot_convergence(result).figure
        self.best_params_ = {param.name: val for param, val in zip(self._dimensions, result.x)}
        return self.estimator.set_params(**params)
