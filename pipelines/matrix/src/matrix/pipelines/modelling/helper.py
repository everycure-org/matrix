import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def get_best_parallel_eval(estimator) -> int:
    # Determine optimal parallelism strategy
    # When estimator uses n_jobs=-1, we need to balance between:
    # 1. Parallel hyperparameter evaluations (outer parallelism)
    # 2. Parallel tree building within each model (inner parallelism)
    #
    # Strategy: Balance parallel evaluations with threads per model to maximize
    # CPU utilization while avoiding resource contention.

    n_cpus = os.cpu_count() - 1 or 1
    logger.info(f"Detected {n_cpus} CPUs for tuning.")
    estimator_n_jobs = getattr(estimator, "n_jobs", 1)
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
        logger.info(f"Searching for parallel evaluations in range: {list(search_range)} with min_threads={min_threads}")

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
        estimator.set_params(n_jobs=threads_per_model)
    elif estimator_n_jobs > 0:
        # Estimator has fixed thread count, evaluate sequentially
        n_parallel_evals = 1
    else:
        # estimator_n_jobs is 1 or None, evaluate in parallel
        n_parallel_evals = -1  # Use all cores for parallel evaluation
    return estimator, n_parallel_evals
