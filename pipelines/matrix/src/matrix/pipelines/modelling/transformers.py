from __future__ import annotations

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
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
        self, head_col, strategy="auto_cv", enabled=True, eta=40, mix_k=5, mix_beta=1.0, eps=1e-6, w_min=1e-3, w_max=20
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

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        counts = X[self.head_col].value_counts()
        freq = counts.reindex(X[self.head_col])

        if not self.enabled:
            w = np.ones_like(freq, dtype=float)
        else:
            match self.strategy:
                case "inverse":
                    w = 1.0 / (freq + self.eps)
                case "shomer":
                    w = np.where(freq < self.eta, 1 + self.mix_beta * self.mix_k, 1)
                case "auto_cv":
                    w = self._weights_auto_cv(freq)
                case "simple_eta":
                    w = np.clip(self.eta / (freq + self.eps), self.w_min, self.w_max)
                case _:
                    raise ValueError

        w = np.clip(w, self.w_min, self.w_max)
        w = w / w.mean()

        self.weight_map_ = dict(zip(X[self.head_col], w))
        self.default_weight_ = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "weight_map_")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        w = X[self.head_col].map(self.weight_map_).fillna(self.default_weight_)
        return pd.DataFrame({"weight": w.values}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(["weight"])

    @staticmethod
    def _to_dataframe(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

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
        target = raw_cv * 1e-3

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
