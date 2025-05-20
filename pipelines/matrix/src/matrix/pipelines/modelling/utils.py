from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_raw_vs_weighted(data: pd.DataFrame):
    head_col = "source"
    degrees = data.groupby(head_col).size()
    weights = data["weight"].to_numpy()

    raw_cnt = data[head_col].map(degrees).to_numpy()
    w_cnt = raw_cnt * weights
    bins = max(10, int(np.sqrt(raw_cnt.size)))
    strategy = "auto cv"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=110)
    ax[0].scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
    ax[0].set(
        xlabel="raw degree",
        ylabel="weighted degree",
        title=f"{strategy} – mapping",
    )

    for vec, col, lab in [
        (raw_cnt, "tab:blue", "raw"),
        (w_cnt, "tab:orange", "weighted"),
    ]:
        sns.histplot(
            vec,
            bins=bins,
            ax=ax[1],
            color=col,
            edgecolor="black",
            alpha=0.30,
            stat="count",
            label=f"{lab} (hist)",
        )
        sns.kdeplot(vec, ax=ax[1], color=col, bw_adjust=1.2, linewidth=2, fill=False, label=f"{lab} KDE")

    ax[1].set(
        xlabel="degree",
        ylabel="entity count",
        title=f"{strategy} – distribution",
    )
    ax[1].legend()

    def _stats(v):
        m = v.mean()
        sd = v.std(ddof=1)
        return [m, np.median(v), sd, sd / m]

    rows = np.vstack([_stats(raw_cnt), _stats(w_cnt)])
    tbl = plt.table(
        cellText=np.round(rows, 3),
        colLabels=["mean", "median", "std", "RSD"],
        rowLabels=["raw", "weighted"],
        bbox=[0.65, -0.24, 0.33, 0.16],
    )
    tbl.auto_set_font_size(True)
    plt.tight_layout()

    return plt.gcf()
