import logging
import os
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)


def get_best_parallel_eval(estimator, n_parallel_trials) -> int:
    # Determine optimal parallelism strategy
    # When estimator uses n_jobs=-1, we need to balance between:
    # 1. Parallel hyperparameter evaluations (outer parallelism)
    # 2. Parallel tree building within each model (inner parallelism)
    #
    # Strategy: Balance parallel evaluations with threads per model to maximize
    # CPU utilization while avoiding resource contention.

    n_cpus = os.cpu_count() or 1
    logger.info(f"Detected {n_cpus} CPUs for tuning.")
    estimator_n_jobs = getattr(estimator, "n_jobs", 1)
    logger.info(f"Estimator n_jobs={estimator_n_jobs}")

    if estimator_n_jobs == -1:
        logger.info("Using n_jobs=-1 for estimator, determining parallel evaluation strategy.")
        # Strategy: Use n_parallel_trials to determine parallel evaluations,
        # and divide CPUs evenly to give each trial proportional threads.
        # This ensures we utilize all CPUs: n_parallel_trials * threads_per_trial = n_cpus
        # With 87 CPUs and n_parallel_trials=3: 3 parallel evaluations × 29 threads = 87 CPUs
        # With 87 CPUs and n_parallel_trials=10: 10 parallel evaluations × 8 threads = 80 CPUs (slight under-utilization)
        # With 87 CPUs and n_parallel_trials=100: 87 parallel evaluations × 1 thread = 87 CPUs (capped at n_cpus)

        # Cap parallel trials at available CPUs (minimum 1 thread per trial)
        n_parallel_evals = min(n_parallel_trials, n_cpus)
        threads_per_model = max(1, n_cpus // n_parallel_evals)

        logger.info(
            f"Using {n_parallel_evals} parallel evaluations with {threads_per_model} threads per model "
            f"(total CPU utilization: {n_parallel_evals * threads_per_model}/{n_cpus})"
        )

        # Update estimator to use proportional threads
        estimator.set_params(n_jobs=threads_per_model)
    elif estimator_n_jobs > 0:
        # Estimator has fixed thread count, evaluate sequentially
        n_parallel_evals = 1
    else:
        # estimator_n_jobs is 1 or None, evaluate in parallel
        n_parallel_evals = -1  # Use all cores for parallel evaluation
    return estimator, n_parallel_evals


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
