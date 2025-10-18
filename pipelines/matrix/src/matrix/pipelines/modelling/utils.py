import logging
from functools import partial
from typing import Any

from matrix_gcp_datasets.spark_utils import detect_gpus

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
