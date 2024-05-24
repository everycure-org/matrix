from functools import partial
from typing import Any, Dict


def inject(dict: Dict[str, Any]):
    pass


def partial_(func: callable, **kwargs):
    """
    Function to wrap partial to enable partial function creation with kwargs.

    Args:
        func: Function to partially apply.
        kwargs: Keyword arguments to partially apply.
    Returns:
        Partially applied function.
    """
    return partial(func, **kwargs)
