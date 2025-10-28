import logging
from typing import List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection._split import _BaseKFold
from skopt import Optimizer
from skopt.plots import plot_convergence
from skopt.space.space import Dimension

from matrix.pipelines.modelling.utils import get_best_parallel_eval

logger = logging.getLogger(__name__)


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
    """Guassian Process based hyperparameter tuner with parallel evaluation support.

    Adaptor class that uses skopt's Optimizer with ask/tell pattern to enable
    parallel evaluation of hyperparameter configurations. When the underlying
    estimator has n_jobs=-1, multiple hyperparameter configurations will be
    evaluated in parallel.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        dimensions: List[Dimension],
        scoring: callable,
        *,
        splitter: _BaseKFold = None,
        n_calls: int = 100,
        n_parallel_trials: int = 1,
    ) -> None:
        """Initialize the tuner.

        Args:
            estimator: sklearn compatible Estimator to tune.
            dimensions: List of dimensions to tune.
            scoring: Scoring function to evaluate the model.
            splitter: Splitter to use for cross-validation.
            n_calls: Number of calls to the objective function.
        """
        self.estimator, self.n_parallel_evals = get_best_parallel_eval(estimator, n_parallel_trials)
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

        def evaluate_model(param_values):
            """Function to evaluate model using the given splitter.

            Function evaluates function using the given splitter and scoring
            functions. When the splitter applies kfold splitting, the scores are
            averaged over the folds.

            FUTURE: Expand invocation of scoring function to inject additional
                columns of the input dataframe, e.g., id's of the drug/disease to
                compute more elaborate metrics, e.g., MRR with synthesized negatives,
                without including these columns as features in the trained model.

            Args:
                param_values: List of parameter values to set on the estimator.
            """
            # Convert parameter values to dictionary
            params_dict = {param.name: val for param, val in zip(self.dimensions, param_values)}
            self.estimator.set_params(**params_dict)

            scores = []
            for train, test in self.splitter.split(X, y):
                X_train_split = X[train]
                self.estimator.fit(X_train_split, y[train])
                X_test_split = X[test]
                y_pred = self.estimator.predict(X_test_split)
                scores.append(self.scoring(y_pred, y[test]))

            return 1.0 - np.average(scores)

        # Create optimizer with Gaussian Process
        optimizer = Optimizer(
            dimensions=self.dimensions,
            base_estimator="GP",
            acq_func="gp_hedge",
            acq_optimizer="lbfgs",
            n_initial_points=min(10, self.n_calls // 2),  # Initial random points
            random_state=self.estimator.random_state if hasattr(self.estimator, "random_state") else None,
        )

        # Run optimization with parallel evaluation
        all_results = []
        for _ in range(self.n_calls):
            # Ask for next point(s) to evaluate
            if self.n_parallel_evals == 1:
                # Sequential: ask for one point at a time
                next_x = optimizer.ask()
                next_y = evaluate_model(next_x)
                result = optimizer.tell(next_x, next_y)
                all_results.append(result)
            else:
                # Parallel: ask for multiple points and evaluate in parallel
                n_points = min(self.n_parallel_evals, self.n_calls - len(all_results))
                next_xs = optimizer.ask(n_points=n_points)

                # Evaluate all points in parallel using joblib
                next_ys = Parallel(n_jobs=self.n_parallel_evals)(delayed(evaluate_model)(x) for x in next_xs)

                # Tell optimizer about all results
                result = optimizer.tell(next_xs, next_ys)
                all_results.append(result)

                # Break if we've reached n_calls
                if len(optimizer.Xi) >= self.n_calls:
                    break

        # Get best parameters
        best_idx = np.argmin(optimizer.yi)
        best_x = optimizer.Xi[best_idx]

        self.convergence_plot = plot_convergence(all_results[-1]).figure

        self.best_params_ = {param.name: val for param, val in zip(self.dimensions, best_x)}

        return self.estimator.set_params(**self.best_params_)
