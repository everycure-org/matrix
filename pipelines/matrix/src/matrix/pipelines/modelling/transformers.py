from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted


class FlatArrayTransformer(FunctionTransformer):
    """sklearn compatible transformer to flatten dataframe with array column in individual columns.

    WARNING: Currently only supports a single input column.
    """

    def __init__(self, prefix: str) -> None:
        """Instantiate the FlatArrayTransformer.

        Args:
            prefix: Prefix to add to the column names.
        """
        self.prefix = prefix
        super().__init__(self._flatten_df_rows)

    @staticmethod
    def _flatten_df_rows(df: pd.DataFrame):
        """Helper function to flat array column into individual columns."""
        return pd.DataFrame(df[df.columns[0]].tolist()).to_numpy()

    def get_feature_names_out(self, input_features=None):
        """Get the feature names of the transformed data.

        Args:
            input_features: Input features to transform.

        Returns:
            List of feature names.
        """
        if input_features.shape[1] > 1:
            raise ValueError("Only one input column is supported.")

        return [f"{self.prefix}{i}" for i in range(len(input_features.iloc[0][0]))]


class WeightingTransformer(BaseEstimator, TransformerMixin):
    """
    Compute per-row sample weights from the degree of *head_col* and
    expose them as a 1-column DataFrame called ``weight``.

    Works out-of-the-box with Kedro pipelines that:
      • call ``fit`` on TRAIN rows only
      • call ``transform`` on any split
      • concatenate the returned DataFrame into the main table
      • optionally pass ``data['weight']`` to ``sample_weight``

    Parameters
    ----------
    head_col : str
        Column in *X* that identifies the head entity.
    strategy : {'inverse', 'shomer', 'auto_cv', 'simple_eta'}, default='auto_cv'
        Weighting rule.
    enabled : bool, default=True
        Disable to return 1.0 everywhere.
    eta, mix_k, mix_beta, eps, w_min, w_max : float, optional
        Hyper-parameters used by the different strategies.
    plot : bool, default=False
        If *True*, save a diagnostic figure for each `transform()` call.
        The file name is ``data/reports/figures/weights/<node_name>.png`` where
        *node_name* comes from the env-var ``KEDRO_NODE_NAME`` (set in a
        `before_node_run` Kedro hook) or `"unknown_node"` if unset.
    keep_original : bool, default=True
        Convenience flag that outside code (e.g. ``apply_transformers``)
        can inspect to decide whether to drop *head_col* after weighting.
        The transformer itself never uses it.
    """

    # -------------------------------------------------- #
    # construction
    # -------------------------------------------------- #
    def __init__(
        self,
        head_col: str,
        strategy: str = "auto_cv",
        enabled: bool = True,
        eta: float = 40.0,
        mix_k: int = 5,
        mix_beta: float = 1.0,
        eps: float = 1e-6,
        w_min: float = 1e-3,
        w_max: float = 20.0,
        plot: bool = True,
        keep_original: bool = True,
    ):
        self.head_col = head_col
        self.strategy = strategy
        self.enabled = enabled
        self.eta = eta
        self.mix_k = mix_k
        self.mix_beta = mix_beta
        self.eps = eps
        self.w_min = w_min
        self.w_max = w_max
        self.plot = plot
        self.keep_original = keep_original

    # -------------------------------------------------- #
    # sklearn API
    # -------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        X = self._to_dataframe(X)

        if self.head_col not in X.columns:
            raise ValueError(f"`head_col='{self.head_col}'` not in columns {list(X.columns)}")

        # minimal metadata sklearn expects
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        shard = os.getenv("SHARD_ID")
        fold = os.getenv("FOLD_ID")
        check_is_fitted(self, "n_features_in_")
        X = self._to_dataframe(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        if not self.enabled:
            weights = np.ones(len(X), dtype=float)
        else:
            degrees = X.groupby(self.head_col).size()
            freq = X[self.head_col].map(degrees)

            match self.strategy:
                case "inverse":
                    weights = self._weights_inverse(freq)
                case "shomer":
                    weights = self._weights_shomer(freq)
                case "auto_cv":
                    weights = self._weights_auto_cv(freq)
                case "simple_eta":
                    weights = np.clip(self.eta / (freq + self.eps), self.w_min, self.w_max)
                    weights /= weights.mean()
                case _:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

            # optional diagnostic plot
            if self.plot:
                self._plot_raw_vs_weighted(
                    raw_cnt=freq.to_numpy(),
                    w_cnt=(weights * freq.to_numpy()),
                    shard=shard,
                    fold=fold,
                )

        return pd.DataFrame({"weight": weights}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(["weight"])

    # -------------------------------------------------- #
    # internal helpers
    # -------------------------------------------------- #
    @staticmethod
    def _to_dataframe(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    # ---------- weighting rules ----------------------- #
    def _weights_shomer(self, cnt):
        w = np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0)
        w = np.clip(w, self.w_min, self.w_max)
        return (w / w.mean()).to_numpy()

    def _weights_inverse(self, cnt):
        w = 1.0 / (cnt + self.eps)
        w = np.clip(w, self.w_min, self.w_max)
        return (w / w.mean()).to_numpy()

    def _weights_auto_cv(self, cnt):
        raw_cv = cnt.std() / cnt.mean()
        target = raw_cv * 1e-3  # fixed tiny CV

        lo, hi = 0.0, 1.0
        for _ in range(300):
            mid = (lo + hi) / 2
            w = self._weights_beta(cnt, mid)
            cv_mid = (w * cnt).std() / (w * cnt).mean()
            lo, hi = (mid, hi) if cv_mid > target else (lo, mid)

        return self._weights_beta(cnt, hi)

    def _weights_beta(self, cnt, beta):
        w = np.clip((cnt + self.eps) ** (-beta), self.w_min, self.w_max)
        return (w / w.mean()).to_numpy()

    # ---------- plotting ------------------------------ #
    def _plot_raw_vs_weighted(self, raw_cnt: np.ndarray, w_cnt: np.ndarray, shard: str, fold: str):
        # shard = os.getenv("SHARD_ID")
        # fold = os.getenv("FOLD_ID")
        bins = max(10, int(np.sqrt(raw_cnt.size)))

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=110)
        ax[0].scatter(raw_cnt, w_cnt, s=18, alpha=0.6, ec="none")
        ax[0].set(
            xlabel="raw degree",
            ylabel="weighted degree",
            title=f"{self.strategy} – mapping",
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
            title=f"{self.strategy} – distribution",
        )
        ax[1].legend()

        # summary stats table
        def _stats(v):
            m = v.mean()
            sd = v.std(ddof=1)
            return [m, np.median(v), sd, sd / m]

        rows = np.vstack([_stats(raw_cnt), _stats(w_cnt)])
        tbl = plt.table(
            cellText=np.round(rows, 3),
            colLabels=["mean", "median", "std", "RSD"],
            rowLabels=["raw", "weighted"],
            loc="bottom",
            bbox=[0.35, -0.32, 0.65, 0.18],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        fig.subplots_adjust(bottom=0.30)
        plt.tight_layout()

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
        catalog.save(
            f"modelling.{shard}.fold_{fold}.reporting.weight_plot",
            plt.gcf(),
            shard=shard,
            fold=fold,
        )

        # plt.savefig(out_dir / f"{safe_name}.png")
        # plt.close()
