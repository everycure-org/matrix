"""Modules with utilities for modelling."""
import numpy as np

from functools import partial
from sklearn.metrics import f1_score


def partial_(func: callable, **kwargs):
    """Function to wrap partial to enable partial function creation with kwargs.

    Args:
        func: Function to partially apply.
        kwargs: Keyword arguments to partially apply.

    Returns:
        Partially applied function.
    """
    return partial(func, **kwargs)


def macro_f1(y_true: np.array, y_pred: np.array) -> float:
    """Returns macro F1 score for multi-class predictions.

    Args:
        y_true: 1d array containing ground truth target values.
        y_pred: 1d array containing estimated targets as returned by a classifier.
    """
    return f1_score(y_true, y_pred, average="macro")
