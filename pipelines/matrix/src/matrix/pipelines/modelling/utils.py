import pandas as pd

from functools import partial


def partial_(func: callable, **kwargs):
    """Function to wrap partial to enable partial function creation with kwargs.

    Args:
        func: Function to partially apply.
        kwargs: Keyword arguments to partially apply.

    Returns:
        Partially applied function.
    """
    return partial(func, **kwargs)


def partial_splits(func: callable, fold: int):
    def func_with_full_splits(data: pd.DataFrame, *args):
        data = data.copy()
        data_fold = data[data.split == fold]
        return partial(func, split=data_fold)(*args)

    return func_with_full_splits
