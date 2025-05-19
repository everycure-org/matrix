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


# ---------- plotting ------------------------------ #
def plot_raw_vs_weighted(data: pd.DataFrame):
    # raw_cnt = X["source"].map(degrees).to_numpy()
    # w_cnt = (X["weight"] * raw_cnt)
    # plot_raw_vs_weighted(X)
    # bins = max(10, int(np.sqrt(raw_cnt.size)))

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=110)
    # ax[0].scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
    # ax[0].set(
    #     xlabel="raw degree",
    #     ylabel="weighted degree",
    #     title=f"{self.strategy} – mapping",
    # )

    # for vec, col, lab in [
    #     (raw_cnt, "tab:blue", "raw"),
    #     (w_cnt, "tab:orange", "weighted"),
    # ]:
    #     sns.histplot(
    #         vec,
    #         bins=bins,
    #         ax=ax[1],
    #         color=col,
    #         edgecolor="black",
    #         alpha=0.30,
    #         stat="count",
    #         label=f"{lab} (hist)",
    #     )
    #     sns.kdeplot(vec, ax=ax[1], color=col, bw_adjust=1.2, linewidth=2, fill=False, label=f"{lab} KDE")

    # ax[1].set(
    #     xlabel="degree",
    #     ylabel="entity count",
    #     title=f"{self.strategy} – distribution",
    # )
    # ax[1].legend()

    # # summary stats table
    # def _stats(v):
    #     m = v.mean()
    #     sd = v.std(ddof=1)
    #     return [m, np.median(v), sd, sd / m]

    # rows = np.vstack([_stats(raw_cnt), _stats(w_cnt)])
    # tbl = plt.table(
    #     cellText=np.round(rows, 3),
    #     colLabels=["mean", "median", "std", "RSD"],
    #     rowLabels=["raw", "weighted"],
    #     loc="bottom",
    #     bbox=[0.35, -0.32, 0.65, 0.18],
    # )
    # tbl.auto_set_font_size(False)
    # tbl.set_fontsize(8)
    # fig.subplots_adjust(bottom=0.30)
    # plt.tight_layout()
    head_col = "source"
    degrees = data.groupby(head_col).size()
    freq = data[head_col].map(degrees).to_numpy()
    weights = data["weight"].to_numpy()
    weighted_deg = freq * weights

    # 2) Build the figure
    bins = max(10, int(np.sqrt(freq.size)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=110)
    ax[0].scatter(freq, weighted_deg, s=18, alpha=0.6, ec="none")
    ax[0].set(xlabel="raw deg", ylabel="weighted deg", title="weight mapping")

    for vec, tag in [(freq, "raw"), (weighted_deg, "weighted")]:
        sns.histplot(vec, bins=bins, stat="count", ax=ax[1], alpha=0.3, label=tag)
        sns.kdeplot(vec, ax=ax[1], bw_adjust=1.2, linewidth=2, fill=False, label=f"{tag} KDE")

    ax[1].set(xlabel="degree", ylabel="count", title="degree distribution")
    ax[1].legend()
    plt.tight_layout()

    return plt.gca()

    # resolve node name for file path
    # node_name = os.getenv("KEDRO_NODE_NAME", "unknown_node")
    # save_path = os.getenv("release_dir", "unknown")
    # safe_name = node_name.replace(" ", "_")
    # out_dir = Path(f"{save_path}/datasets/reports/figures/weights")
    # out_dir = Path("gs://mtrx-us-central1-hub-dev-storage/kedro/data/tests/v0.4.5/datasets/reports/figures/weights")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # catalog.save("weight_diagnostic_plot", plt.gcf(), node_name=os.getenv("KEDRO_NODE_NAME").replace(" ", "_"))

    # from kedro.framework.context import get_current_context

    # catalog = get_current_context().catalog
    # catalog.save(
    #     f"modelling.{shard}.fold_{fold}.reporting.weight_plot",
    #     plt.gcf(),
    #     shard=shard,
    #     fold=fold,
    # )

    # plt.savefig(out_dir / f"{safe_name}.png")
    # plt.close()
