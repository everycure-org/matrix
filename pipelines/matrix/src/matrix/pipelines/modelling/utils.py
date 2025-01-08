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
