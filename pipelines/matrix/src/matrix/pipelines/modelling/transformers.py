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


class WeightingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        head_col,
        tail_col,
        label_col="y",
        pos_label=1,
        budget_pos=1.0,
        budget_neg=1.0,
        iters=20,
        eps=1e-6,
        w_min=0.25,
        w_max=4,
        normalize="per_class",
        output_name="weight",
        default_class="neg",
        rho=0.3,
        clip_factors=False,
        a_min=0.5,
        a_max=2.0,
    ):
        self.head_col = head_col
        self.tail_col = tail_col
        self.label_col = label_col
        self.pos_label = pos_label
        self.budget_pos = budget_pos
        self.budget_neg = budget_neg
        self.iters = iters
        self.eps = eps
        self.w_min = w_min
        self.w_max = w_max
        self.normalize = normalize
        self.output_name = output_name
        self.default_class = default_class
        self.rho = rho
        self.clip_factors = clip_factors
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = X[self.label_col] if (y is None and self.label_col in X) else pd.Series(y, index=X.index)
        pos = y == self.pos_label

        def _sinkhorn(df, budget):
            if df.empty:
                return {}, {}
            u = df[self.head_col].unique()
            v = df[self.tail_col].unique()
            a = pd.Series(1.0, index=u, dtype=float)
            b = pd.Series(1.0, index=v, dtype=float)
            Lmin, Lmax = np.log(self.a_min), np.log(self.a_max)

            for _ in range(self.iters):
                bv = df[self.tail_col].map(b).to_numpy()
                sum_b_by_u = (
                    pd.DataFrame({self.head_col: df[self.head_col].to_numpy(), "bv": bv})
                    .groupby(self.head_col)["bv"]
                    .sum()
                )
                a_new = (budget / np.maximum(sum_b_by_u, self.eps)).reindex(a.index).fillna(1.0)

                La = np.log(np.maximum(a.to_numpy(), self.eps))
                Lanew = np.log(np.maximum(a_new.to_numpy(), self.eps))
                La = (1 - self.rho) * La + self.rho * Lanew
                if self.clip_factors:
                    La = np.clip(La, Lmin, Lmax)
                a = pd.Series(np.exp(La), index=a.index)

                au = df[self.head_col].map(a).to_numpy()
                sum_a_by_v = (
                    pd.DataFrame({self.tail_col: df[self.tail_col].to_numpy(), "au": au})
                    .groupby(self.tail_col)["au"]
                    .sum()
                )
                b_new = (budget / np.maximum(sum_a_by_v, self.eps)).reindex(b.index).fillna(1.0)

                Lb = np.log(np.maximum(b.to_numpy(), self.eps))
                Lbnew = np.log(np.maximum(b_new.to_numpy(), self.eps))
                Lb = (1 - self.rho) * Lb + self.rho * Lbnew
                if self.clip_factors:
                    Lb = np.clip(Lb, Lmin, Lmax)
                b = pd.Series(np.exp(Lb), index=b.index)

            return a.to_dict(), b.to_dict()

        a_pos, b_pos = _sinkhorn(X.loc[pos, [self.head_col, self.tail_col]], self.budget_pos)
        a_neg, b_neg = _sinkhorn(X.loc[~pos, [self.head_col, self.tail_col]], self.budget_neg)

        self.a_pos_, self.b_pos_ = a_pos, b_pos
        self.a_neg_, self.b_neg_ = a_neg, b_neg
        self.feature_names_in_ = np.array(list(X.columns))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_in_")
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if y is None and self.label_col in X:
            y = X[self.label_col]
        elif y is None:
            use_pos = self.default_class == "pos"
            y = pd.Series(self.pos_label if use_pos else object(), index=X.index)
        else:
            y = pd.Series(y, index=X.index)

        pos = y == self.pos_label

        def _weights(df, a_map, b_map):
            a = df[self.head_col].map(a_map).fillna(1.0).to_numpy()
            b = df[self.tail_col].map(b_map).fillna(1.0).to_numpy()
            return a * b

        w = np.empty(len(X), dtype=float)
        w[pos] = _weights(X.loc[pos], self.a_pos_, self.b_pos_)
        w[~pos] = _weights(X.loc[~pos], self.a_neg_, self.b_neg_)

        # w = np.clip(w, self.w_min, self.w_max)
        if self.normalize == "per_class":
            if pos.any():
                w[pos] /= max(w[pos].mean(), self.eps)
            if (~pos).any():
                w[~pos] /= max(w[~pos].mean(), self.eps)
        else:
            w /= max(w.mean(), self.eps)

        return np.asarray(w).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_name])


# class WeightingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         head_col,
#         strategy="auto_cv",
#         enabled=True,
#         eta=40,
#         mix_k=5,
#         mix_beta=1.0,
#         eps=1e-6,
#         w_min=1e-3,
#         w_max=20.0,
#         *,
#         per_class=False,
#         pos_label=1,
#         normalize="per_class",
#         default_class="neg",
#     ):
#         self.head_col = head_col
#         self.strategy = strategy
#         self.enabled = enabled
#         self.eta = eta
#         self.mix_k = mix_k
#         self.mix_beta = mix_beta
#         self.eps = eps
#         self.w_min = w_min
#         self.w_max = w_max

#         self.per_class = per_class
#         self.label_col = "y"
#         self.pos_label = pos_label
#         self.normalize = normalize
#         self.default_class = default_class

#         self.class_budget_pos = 1.0
#         self.class_budget_neg = 1.0
#         self.enforce_budget = True
#         self.beta = None
#         self.log_clip = True

#     def fit(self, X, y=None):
#         X = self._ensure_dataframe(X)
#         y_series = self._get_y_series(X, y)

#         if not self.enabled:
#             w = np.ones(len(X), dtype=float)
#             self._store_maps_from_row_weights(X, y_series, w)
#             self.n_features_in_ = X.shape[1]
#             return self

#         if not self.per_class:
#             cnt = X[self.head_col].value_counts()
#             base_w = self._weights_from_counts(cnt)
#             w = X[self.head_col].map(base_w).fillna(1.0).to_numpy()
#             w = self._clip(w)
#             w = self._normalize(w, y_series)
#             self._store_maps_from_row_weights(X, y_series, w)
#             self.n_features_in_ = X.shape[1]
#             return self

#         if y_series is None:
#             raise ValueError("per_class requires labels")

#         pos_mask = y_series == self.pos_label
#         pos_mask = pos_mask.reindex(X.index, fill_value=False)
#         neg_mask = ~pos_mask

#         cnt_pos = X.loc[pos_mask, self.head_col].value_counts()
#         cnt_neg = X.loc[neg_mask, self.head_col].value_counts()

#         w_map_pos_raw = self._weights_from_counts(cnt_pos)
#         w_map_neg_raw = self._weights_from_counts(cnt_neg)

#         if self.strategy == "marginal":
#             w_map_pos_raw = self.class_budget_pos * w_map_pos_raw
#             w_map_neg_raw = self.class_budget_neg * w_map_neg_raw

#         w = np.empty(len(X), dtype=float)
#         w[pos_mask] = X.loc[pos_mask, self.head_col].map(w_map_pos_raw).fillna(1.0).to_numpy()
#         w[neg_mask] = X.loc[neg_mask, self.head_col].map(w_map_neg_raw).fillna(1.0).to_numpy()

#         w = self._log_clip(w)

#         if self.enforce_budget and pos_mask.any():
#             w[pos_mask] = self._apply_budget(
#                 X.loc[pos_mask, [self.head_col]], w[pos_mask], self.head_col, self.class_budget_pos
#             )
#         if self.enforce_budget and neg_mask.any():
#             w[neg_mask] = self._apply_budget(
#                 X.loc[neg_mask, [self.head_col]], w[neg_mask], self.head_col, self.class_budget_neg
#             )

#         if self.normalize == "per_class":
#             if pos_mask.any():
#                 w[pos_mask] /= max(w[pos_mask].mean(), self.eps)
#             if neg_mask.any():
#                 w[neg_mask] /= max(w[neg_mask].mean(), self.eps)
#         else:
#             w /= max(w.mean(), self.eps)

#         if pos_mask.any():
#             pos_scale = 1.0
#             w_map_pos = w_map_pos_raw.copy()
#             mu_pos = X.loc[pos_mask, self.head_col].map(w_map_pos).fillna(1.0).mean()
#             if mu_pos > 0:
#                 w_map_pos = w_map_pos / mu_pos
#         else:
#             w_map_pos = pd.Series(dtype=float)

#         if neg_mask.any():
#             w_map_neg = w_map_neg_raw.copy()
#             mu_neg = X.loc[neg_mask, self.head_col].map(w_map_neg).fillna(1.0).mean()
#             if mu_neg > 0:
#                 w_map_neg = w_map_neg / mu_neg
#         else:
#             w_map_neg = pd.Series(dtype=float)

#         self.weight_map_pos_ = w_map_pos.to_dict()
#         self.weight_map_neg_ = w_map_neg.to_dict()
#         self.default_weight_pos_ = 1.0
#         self.default_weight_neg_ = 1.0

#         self.n_features_in_ = X.shape[1]
#         return self

#     def transform(self, X, y=None):
#         check_is_fitted(self, "n_features_in_")
#         X = self._to_dataframe(X)
#         y_series = self._get_y_series(X, y)

#         if hasattr(self, "weight_map_pos_") and self.per_class:
#             if y_series is None:
#                 use_pos = self.default_class == "pos"
#                 w = X[self.head_col].map(self.weight_map_pos_ if use_pos else self.weight_map_neg_).fillna(1.0)
#             else:
#                 pos_mask = y_series == self.pos_label
#                 w = pd.Series(index=X.index, dtype=float)
#                 w[pos_mask] = X.loc[pos_mask, self.head_col].map(self.weight_map_pos_).fillna(1.0)
#                 w[~pos_mask] = X.loc[~pos_mask, self.head_col].map(self.weight_map_neg_).fillna(1.0)
#             w = self._clip(w.to_numpy())
#             return pd.DataFrame({"weight": w}, index=X.index)

#         # single-map mode
#         check_is_fitted(self, "weight_map_")
#         w = X[self.head_col].map(self.weight_map_).fillna(self.default_weight_)
#         w = self._clip(w.to_numpy())
#         return pd.DataFrame({"weight": w.values}, index=X.index)

#     def get_feature_names_out(self, input_features=None):
#         return np.array(["weight"])

#     @staticmethod
#     def _to_dataframe(X):
#         return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

#     def _ensure_dataframe(self, X):
#         if isinstance(X, pd.Series):
#             X = X.to_frame(name=self.head_col if isinstance(self.head_col, str) else "feature")
#         elif not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)

#         if self.head_col not in X.columns:
#             if X.shape[1] == 1:
#                 X = X.copy()
#                 X.columns = [self.head_col]
#             else:
#                 raise KeyError(f"head_col '{self.head_col}' not in X.columns: {list(X.columns)[:8]}...")
#         return X

#     def _log_clip(self, w):
#         if not self.log_clip:
#             return np.clip(w, self.w_min, self.w_max)
#         lw = np.log(np.maximum(w, self.eps))
#         lw = np.clip(lw, np.log(self.w_min), np.log(self.w_max))
#         return np.exp(lw)

#     def _weights_from_counts(self, cnt: pd.Series) -> pd.Series:
#         strat = self.strategy
#         if self.beta is not None:
#             # explicit tempered inverse: w ∝ cnt^{-beta}
#             w = (np.maximum(cnt, self.eps) ** (-self.beta)).astype(float)
#         elif strat == "inverse":
#             w = 1.0 / (cnt + self.eps)
#         elif strat == "inverse_sqrt":
#             w = 1.0 / np.sqrt(cnt + self.eps)
#         elif strat == "marginal":  # NEW: per-entity budget / count
#             # budget set later per-class; return just 1/cnt shape
#             w = 1.0 / (cnt + self.eps)
#         elif strat == "shomer":
#             w = pd.Series(np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0), index=cnt.index)
#         elif strat == "auto_cv":
#             w = pd.Series(self._weights_auto_cv(cnt), index=cnt.index)
#         elif strat == "simple_eta":
#             w = pd.Series(np.clip(self.eta / (cnt + self.eps), self.w_min, self.w_max), index=cnt.index)
#         else:
#             raise ValueError(f"Unknown strategy: {strat}")

#         if isinstance(w, np.ndarray):
#             w = pd.Series(w, index=cnt.index)
#         # clip symmetrically
#         w = pd.Series(self._log_clip(w.to_numpy()), index=cnt.index)
#         return w

#     def _apply_budget(self, X_sub: pd.DataFrame, w_rows: np.ndarray, entity_col: str, budget: float) -> np.ndarray:
#         """
#         Per-entity exact budget enforcement on the already-mapped row weights.
#         Scales rows per entity so sum_rows(entity) == budget (up to a class-wide scalar).
#         """
#         g = pd.DataFrame({entity_col: X_sub[entity_col].to_numpy(), "w": w_rows})
#         sums = g.groupby(entity_col)["w"].sum()
#         # avoid zero-division for unseen entities
#         scale_per_entity = (budget / np.maximum(sums, self.eps)).to_dict()
#         scales = X_sub[entity_col].map(scale_per_entity).to_numpy()
#         return w_rows * scales

#     def _get_y_series(self, X, y):
#         if self.label_col is not None and isinstance(X, pd.DataFrame) and self.label_col in X.columns:
#             s = X[self.label_col]
#             return s if isinstance(s, pd.Series) else pd.Series(s, index=X.index)

#         if y is None:
#             return None

#         if np.isscalar(y) or (hasattr(y, "ndim") and getattr(y, "ndim", 0) == 0):
#             return pd.Series(np.full(len(X), y), index=X.index)

#         y_arr = np.asarray(y)
#         if y_arr.ndim == 1 and len(y_arr) == len(X):
#             return pd.Series(y_arr, index=X.index)

#         raise ValueError(f"y must be None, a scalar, or 1D array of length {len(X)}; got {getattr(y_arr,'shape',None)}")

#     def _clip(self, w):
#         return np.clip(w, self.w_min, self.w_max)

#     def _weights_from_counts(self, cnt: pd.Series) -> pd.Series:
#         if self.strategy == "inverse":
#             w = 1.0 / (cnt + self.eps)
#         elif self.strategy == "shomer":
#             w = pd.Series(np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0), index=cnt.index)
#         elif self.strategy == "auto_cv":
#             w = pd.Series(self._weights_auto_cv(cnt), index=cnt.index)
#         elif self.strategy == "simple_eta":
#             w = pd.Series(np.clip(self.eta / (cnt + self.eps), self.w_min, self.w_max), index=cnt.index)
#         else:
#             raise ValueError(f"Unknown strategy: {self.strategy}")

#         if isinstance(w, np.ndarray):
#             w = pd.Series(w, index=cnt.index)
#         w = w.clip(lower=self.w_min, upper=self.w_max)
#         return w

#     def _weights_shomer(self, cnt):
#         w = np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0)
#         w = np.clip(w, self.w_min, self.w_max)
#         return (w / w.mean()).to_numpy()

#     def _weights_inverse(self, cnt):
#         w = 1.0 / (cnt + self.eps)
#         w = np.clip(w, self.w_min, self.w_max)
#         return (w / w.mean()).to_numpy()

#     def _weights_auto_cv(self, cnt):
#         raw_cv = cnt.std() / max(cnt.mean(), self.eps)
#         target = raw_cv * 1e-3
#         lo, hi = 0.0, 1.0
#         for _ in range(300):
#             mid = (lo + hi) / 2
#             w = self._weights_beta(cnt, mid)
#             cv_mid = (w * cnt).std() / max((w * cnt).mean(), self.eps)
#             lo, hi = (mid, hi) if cv_mid > target else (lo, mid)
#         return self._weights_beta(cnt, hi)

#     def _weights_beta(self, cnt, beta):
#         w = np.clip((cnt + self.eps) ** (-beta), self.w_min, self.w_max)
#         return (w / max(w.mean(), self.eps)).to_numpy()
