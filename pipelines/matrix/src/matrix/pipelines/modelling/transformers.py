from __future__ import annotations

import numpy as np
import pandas as pd
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


class PairwiseEmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Degree-neutral pairwise features from two embedding vectors.

    Emits: [pair_cos, pair_l2, pair_angle?, had_mean/std/min/max, diff_mean/std/min/max]
    """

    def __init__(
        self, source_col="source_embedding", target_col="target_embedding", eps: float = 1e-12, add_angle: bool = True
    ):
        self.source_col = source_col
        self.target_col = target_col
        self.eps = eps
        self.add_angle = add_angle
        self._feature_names = None

    def fit(self, X, y=None):
        base = ["pair_cos", "pair_l2"]
        if self.add_angle:
            base.append("pair_angle")
        base += [
            "had_mean",
            "had_std",
            "had_min",
            "had_max",
            "diff_mean",
            "diff_std",
            "diff_min",
            "diff_max",
        ]
        self._feature_names = base
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        u = np.stack(df[self.source_col].to_numpy())
        v = np.stack(df[self.target_col].to_numpy())

        # unit-normalize to remove norm effects
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + self.eps)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + self.eps)

        cos = (u * v).sum(axis=1)
        l2 = np.linalg.norm(u - v, axis=1)

        had = u * v
        diff = np.abs(u - v)

        def aggr(M):
            return np.c_[M.mean(1), M.std(1), M.min(1), M.max(1)]

        h = aggr(had)
        d = aggr(diff)

        cols = [cos, l2]
        if self.add_angle:
            ang = np.arccos(np.clip(cos, -1.0, 1.0))
            cols.append(ang)
        cols += [h, d]

        out = np.c_[*cols].astype(np.float32)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_feature_names")
        return np.array(self._feature_names, dtype=object)


class WeightingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        head_col,
        strategy="auto_cv",
        enabled=True,
        eta=40,
        mix_k=5,
        mix_beta=1.0,
        eps=1e-6,
        w_min=1e-3,
        w_max=20.0,
        *,
        per_class=False,
        pos_label=1,
        normalize="per_class",
        default_class="neg",
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

        self.per_class = per_class
        self.label_col = "y"
        self.pos_label = pos_label
        self.normalize = normalize
        self.default_class = default_class

        self.class_budget_pos = 1.0
        self.class_budget_neg = 1.0
        self.enforce_budget = False
        self.beta = None
        self.log_clip = False

    def fit(self, X, y=None):
        X = self._ensure_dataframe(X)
        y_series = self._get_y_series(X, y)

        if not self.enabled:
            self._init_identity_maps()
            self.n_features_in_ = X.shape[1]
            return self

        if not self.per_class:
            cnt_all = X[self.head_col].value_counts()
            w_map = self._weights_from_counts(cnt_all)
            mu = X[self.head_col].map(w_map).fillna(1.0).mean()
            if mu and mu > 0:
                w_map = (w_map / mu).clip(lower=self.w_min, upper=self.w_max)
            self.weight_map_ = w_map.to_dict()
            self.default_weight_ = 1.0
            self.n_features_in_ = X.shape[1]
            return self

        if y_series is None:
            raise ValueError("per_class=True requires labels ('y' in X or provided as y).")

        pos_mask = (y_series == self.pos_label).reindex(X.index, fill_value=False)
        neg_mask = ~pos_mask

        cnt_pos = X.loc[pos_mask, self.head_col].value_counts()
        cnt_neg = X.loc[neg_mask, self.head_col].value_counts()

        w_map_pos = self._weights_from_counts(cnt_pos)
        w_map_neg = self._weights_from_counts(cnt_neg)

        # Normalization
        if self.normalize == "per_class":
            mu_pos = X.loc[pos_mask, self.head_col].map(w_map_pos).fillna(1.0).mean() if pos_mask.any() else 1.0
            mu_neg = X.loc[neg_mask, self.head_col].map(w_map_neg).fillna(1.0).mean() if neg_mask.any() else 1.0
            if mu_pos and mu_pos > 0:
                w_map_pos = (w_map_pos / mu_pos).clip(lower=self.w_min, upper=self.w_max)
            if mu_neg and mu_neg > 0:
                w_map_neg = (w_map_neg / mu_neg).clip(lower=self.w_min, upper=self.w_max)
        else:
            w_pos_rows = (
                X.loc[pos_mask, self.head_col].map(w_map_pos).fillna(1.0).to_numpy() if pos_mask.any() else np.array([])
            )
            w_neg_rows = (
                X.loc[neg_mask, self.head_col].map(w_map_neg).fillna(1.0).to_numpy() if neg_mask.any() else np.array([])
            )
            concat = (
                np.concatenate([w_pos_rows, w_neg_rows]) if (w_pos_rows.size or w_neg_rows.size) else np.array([1.0])
            )
            mu_all = float(np.mean(concat)) if concat.size else 1.0
            if mu_all and mu_all > 0:
                w_map_pos = (w_map_pos / mu_all).clip(lower=self.w_min, upper=self.w_max)
                w_map_neg = (w_map_neg / mu_all).clip(lower=self.w_min, upper=self.w_max)

        self.weight_map_pos_ = w_map_pos.to_dict()
        self.weight_map_neg_ = w_map_neg.to_dict()
        self.default_weight_pos_ = 1.0
        self.default_weight_neg_ = 1.0

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_in_")
        X = self._to_dataframe(X)
        if self.per_class:
            y_series = self._get_y_series(X, y)
            if y_series is None:
                use_pos = self.default_class == "pos"
                w_series = X[self.head_col].map(self.weight_map_pos_ if use_pos else self.weight_map_neg_).fillna(1.0)
            else:
                pos_mask = (y_series == self.pos_label).reindex(X.index, fill_value=False)
                w_series = pd.Series(index=X.index, dtype=float)
                w_series[pos_mask] = X.loc[pos_mask, self.head_col].map(self.weight_map_pos_).fillna(1.0)
                w_series[~pos_mask] = X.loc[~pos_mask, self.head_col].map(self.weight_map_neg_).fillna(1.0)
            w = self._clip(w_series.to_numpy())
            return pd.DataFrame({"weight": w}, index=X.index)

        check_is_fitted(self, ["weight_map_", "default_weight_"])
        w = X[self.head_col].map(self.weight_map_).fillna(self.default_weight_)
        w = self._clip(np.asarray(w, dtype=float))
        return pd.DataFrame({"weight": w}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(["weight"])

    # ----------------- helpers -----------------

    @staticmethod
    def _to_dataframe(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _ensure_dataframe(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame(name=self.head_col if isinstance(self.head_col, str) else "feature")
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.head_col not in X.columns:
            if X.shape[1] == 1:
                X = X.copy()
                X.columns = [self.head_col]
            else:
                raise KeyError(f"head_col '{self.head_col}' not in X.columns: {list(X.columns)[:8]}...")
        return X

    def _get_y_series(self, X, y):
        if self.label_col is not None and isinstance(X, pd.DataFrame) and self.label_col in X.columns:
            s = X[self.label_col]
            return s if isinstance(s, pd.Series) else pd.Series(s, index=X.index)
        if y is None:
            return None
        if np.isscalar(y) or (hasattr(y, "ndim") and getattr(y, "ndim", 0) == 0):
            return pd.Series(np.full(len(X), y), index=X.index)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1 and len(y_arr) == len(X):
            return pd.Series(y_arr, index=X.index)
        raise ValueError(
            f"y must be None, a scalar, or 1D array of length {len(X)}; got {getattr(y_arr, 'shape', None)}"
        )

    def _clip(self, w):
        return np.clip(w, self.w_min, self.w_max)

    def _init_identity_maps(self):
        if self.per_class:
            self.weight_map_pos_ = {}
            self.weight_map_neg_ = {}
            self.default_weight_pos_ = 1.0
            self.default_weight_neg_ = 1.0
        else:
            self.weight_map_ = {}
            self.default_weight_ = 1.0

    def _weights_from_counts(self, cnt: pd.Series) -> pd.Series:
        """
        Map entity counts -> weights for the current strategy.
        Returns a Series aligned to cnt.index.
        """
        strat = str(self.strategy).lower()
        if strat == "inverse":
            w = 1.0 / (cnt + self.eps)
        elif strat == "inverse_sqrt":
            w = 1.0 / np.sqrt(cnt + self.eps)
        elif strat == "shomer":
            w = np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0)
        elif strat == "auto_cv":
            w = self._weights_auto_cv(cnt)
        elif strat == "simple_eta":
            w = np.clip(self.eta / (cnt + self.eps), self.w_min, self.w_max)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if isinstance(w, np.ndarray):
            w = pd.Series(w, index=cnt.index)
        else:
            w = pd.Series(w, index=cnt.index)
        w = w.clip(lower=self.w_min, upper=self.w_max)
        return w

    def _weights_auto_cv(self, cnt: pd.Series) -> np.ndarray:
        raw_cv = cnt.std() / max(cnt.mean(), self.eps)
        target = raw_cv * 1e-3
        lo, hi = 0.0, 1.0
        for _ in range(300):
            mid = (lo + hi) / 2
            w = self._weights_beta(cnt, mid)
            cv_mid = (w * cnt).std() / max((w * cnt).mean(), self.eps)
            lo, hi = (mid, hi) if cv_mid > target else (lo, mid)
        return self._weights_beta(cnt, hi)

    def _weights_beta(self, cnt: pd.Series, beta: float) -> np.ndarray:
        w = (np.asarray(cnt, dtype=float) + self.eps) ** (-beta)
        w = np.clip(w, self.w_min, self.w_max)
        denom = float(w.mean()) if w.size else self.eps
        denom = max(denom, self.eps)
        return w / denom

    def _weights_shomer(self, cnt):
        w = np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0)
        w = np.clip(w, self.w_min, self.w_max)
        return (w / w.mean()).to_numpy()

    def _weights_inverse(self, cnt):
        w = 1.0 / (cnt + self.eps)
        w = np.clip(w, self.w_min, self.w_max)
        return (w / w.mean()).to_numpy()
