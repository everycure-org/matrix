import logging
import os
from functools import partial
from typing import Any

import numpy as np
from matrix_gcp_datasets.spark_utils import detect_gpus

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


try:  # Cupy is optional; only needed when running estimators on CUDA
    if detect_gpus() > 0:
        import cupy as cp
    else:
        raise ImportError("No GPUs detected")
except ImportError:  # pragma: no cover - executed only on CPU-only setups
    cp = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def partial_(func: callable, **kwargs):
    """Function to wrap partial to enable partial function creation with kwargs.

    Args:
        func: Function to partially apply.
        kwargs: Keyword arguments to partially apply.

    Returns:
        Partially applied function.
    """
    return partial(func, **kwargs)


def partial_fold(func: callable, fold: int, arg_name: str = "data"):
    """Creates a partial function that takes a full dataset and operates on a specific fold.

    NOTE: When applying this function in a Kedro node, the inputs must be explicitly stated as a dictionary, not a list.

    Args:
        func: Function operating on a specific fold of the data.
        fold: The fold number to filter the data on.
        arg_name: The name of the argument to filter the data on.

    Returns:
        A function that takes full data and additional arguments/kwargs and applies the original
        function to the specified fold.
    """

    def func_with_full_splits(**kwargs):
        data = kwargs[arg_name]
        data_fold = data[data["fold"] == fold]
        kwargs[arg_name] = data_fold
        return func(**kwargs)

    return func_with_full_splits


def estimator_uses_cuda(estimator: Any) -> bool:
    """Return True when the estimator is configured to run on CUDA."""

    if not hasattr(estimator, "get_params"):
        return False

    params = estimator.get_params(deep=False)
    device = params.get("device")
    tree_method = params.get("tree_method")
    predictor = params.get("predictor")
    gpu_id = params.get("gpu_id")

    if isinstance(device, str) and device.lower() == "cuda":
        return True
    if isinstance(tree_method, str) and tree_method.startswith("gpu"):
        return True
    if isinstance(predictor, str) and predictor.startswith("gpu"):
        return True
    if gpu_id is not None and gpu_id not in (-1, None):
        return True

    return False


def to_estimator_device(array: Any, estimator: Any) -> Any:
    """Move array to the estimator's preferred device when CUDA is requested."""
    try:
        if not estimator_uses_cuda(estimator):
            return array

        if cp is None:
            logger.warning(
                "Estimator %s configured for CUDA but CuPy is not available; falling back to CPU arrays.",
                estimator,
            )
            return array

        if isinstance(array, cp.ndarray):  # Already on GPU
            return array

        return cp.asarray(array)
    except Exception as e:
        logger.warning(
            "Failed to move array to estimator's device: %s; falling back to CPU arrays.",
            e,
        )
        return array


def to_cpu(array: Any) -> Any:
    """Move array from GPU to CPU if it's a CuPy array.

    Args:
        array: Array to move to CPU (can be CuPy ndarray, NumPy ndarray, or other array-like).

    Returns:
        NumPy array or the original array if it's not a CuPy array.
    """
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    return array
