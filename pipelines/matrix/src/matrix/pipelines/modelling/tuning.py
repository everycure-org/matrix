import logging
from typing import List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection._split import _BaseKFold
from skopt import Optimizer
from skopt.plots import plot_convergence
from skopt.space.space import Dimension

from .utils import to_estimator_device

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

    NOTE: The parallel evaluation implementation was partially generated using AI assistance.
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
                X_train_split = to_estimator_device(X[train], self.estimator)
                self.estimator.fit(X_train_split, y[train])
                X_test_split = to_estimator_device(X[test], self.estimator)
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

        # Determine optimal parallelism strategy
        # When estimator uses n_jobs=-1, we need to balance between:
        # 1. Parallel hyperparameter evaluations (outer parallelism)
        # 2. Parallel tree building within each model (inner parallelism)
        #
        # Strategy: Balance parallel evaluations with threads per model to maximize
        # CPU utilization while avoiding resource contention.
        import os

        n_cpus = os.cpu_count() - 1 or 1
        logger.info(f"Detected {n_cpus} CPUs for tuning.")
        estimator_n_jobs = getattr(self.estimator, "n_jobs", 1)
        logger.info(f"Estimator n_jobs={estimator_n_jobs}")

        if estimator_n_jobs == -1:
            logger.info("Using n_jobs=-1 for estimator, determining parallel evaluation strategy.")
            # Balance: evaluate multiple configs in parallel, give each proportional threads
            # For large CPU counts, use more parallel evaluations to better utilize resources
            # Formula: Use factors that divide CPU count well, with preference for 3-8 parallel evals
            #
            # Strategy for different CPU counts:
            # - 88 CPUs: 4 parallel × 22 threads = 88
            # - 111 CPUs: 3 parallel × 37 threads = 111
            # - 128 CPUs: 8 parallel × 16 threads = 128

            # Determine acceptable range for parallel evaluations based on CPU count
            if n_cpus <= 32:
                search_range = range(2, 5)  # 2-4 parallel
                min_threads = 4
            elif n_cpus <= 64:
                search_range = range(3, 7)  # 3-6 parallel
                min_threads = 8
            else:
                search_range = range(3, 9)  # 3-8 parallel
                min_threads = 12
            logger.info(
                f"Searching for parallel evaluations in range: {list(search_range)} with min_threads={min_threads}"
            )

            # Find the divisor that gives best CPU utilization while maintaining
            # good thread count per model for XGBoost performance
            best_divisor = None
            best_efficiency = 0
            best_threads = 0

            for candidate in search_range:
                threads = n_cpus // candidate
                efficiency = (candidate * threads) / n_cpus

                # Only consider if threads per model is sufficient for good performance
                if threads >= min_threads:
                    # Prefer better efficiency, but also consider thread count
                    # (higher threads per model often performs better)
                    score = efficiency + (threads / n_cpus) * 0.1  # Slight bonus for more threads

                    if best_divisor is None or score > best_efficiency:
                        best_divisor = candidate
                        best_efficiency = score
                        best_threads = threads

            # Fallback if no good option found (shouldn't happen with current ranges)
            if best_divisor is None:
                best_divisor = max(2, int(np.sqrt(n_cpus)))
                best_threads = max(1, n_cpus // best_divisor)

            n_parallel_evals = best_divisor
            threads_per_model = best_threads

            # Update estimator to use proportional threads
            self.estimator.set_params(n_jobs=threads_per_model)
        elif estimator_n_jobs > 0:
            # Estimator has fixed thread count, evaluate sequentially
            n_parallel_evals = 1
        else:
            # estimator_n_jobs is 1 or None, evaluate in parallel
            n_parallel_evals = -1  # Use all cores for parallel evaluation

        # Run optimization with parallel evaluation
        all_results = []
        for _ in range(self.n_calls):
            # Ask for next point(s) to evaluate
            if n_parallel_evals == 1:
                # Sequential: ask for one point at a time
                next_x = optimizer.ask()
                next_y = evaluate_model(next_x)
                result = optimizer.tell(next_x, next_y)
                all_results.append(result)
            else:
                # Parallel: ask for multiple points and evaluate in parallel
                n_points = min(n_parallel_evals, self.n_calls - len(all_results))
                next_xs = optimizer.ask(n_points=n_points)

                # Evaluate all points in parallel using joblib
                next_ys = Parallel(n_jobs=n_parallel_evals)(delayed(evaluate_model)(x) for x in next_xs)

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
