import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


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
    """Transformer that computes sample weights based on node degrees."""

    def __init__(
        self,
        head_col: str,
        strategy: str = "shomer",
        eta: int = 40,
        mix_k: int = 5,
        mix_beta: float = 1.0,
        eps: float = 1e-6,
        w_min: float = 1e-3,
        w_max: float = 20.0,
    ):
        """Initialize weighting transformer.

        Args:
            head_col: Column to compute degrees on ('source' or 'target')
            strategy: Weighting strategy ('shomer', 'inverse', or 'auto_cv')
            eta: Degree threshold for rare nodes (Shomer strategy)
            mix_k: Mixing factor k (Shomer strategy)
            mix_beta: Mixing beta parameter (Shomer strategy)
            eps: Small constant to avoid division by zero
            w_min: Minimum weight value
            w_max: Maximum weight value
        """
        self.head_col = head_col
        self.strategy = strategy
        self.eta = eta
        self.mix_k = mix_k
        self.mix_beta = mix_beta
        self.eps = eps
        self.w_min = w_min
        self.w_max = w_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Compute weights based on node degrees.

        Args:
            X: Input DataFrame with head_col and y column

        Returns:
            Sample weights array
        """
        degrees = X.groupby(self.head_col).size().to_dict()
        node_degrees = np.array([degrees[x] for x in X[self.head_col]])

        if self.strategy == "shomer":
            weights = self._weights_shomer(node_degrees)
        elif self.strategy == "inverse":
            weights = self._weights_inverse(node_degrees)
        elif self.strategy == "auto_cv":
            weights = self._weights_auto_cv(node_degrees)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.strategy}")

        return weights

    def _weights_shomer(self, cnt):
        """Shomer et al piece-wise constant reweighting."""
        w = np.where(cnt < self.eta, 1.0 + self.mix_beta * self.mix_k, 1.0)
        w = np.clip(w, self.w_min, self.w_max)
        return w / w.mean()

    def _weights_inverse(self, cnt):
        """Simple inverse degree weighting."""
        w = 1.0 / (cnt + self.eps)
        w = np.clip(w, self.w_min, self.w_max)
        return w / w.mean()

    def _weights_auto_cv(self, cnt):
        """Auto CV ratio weighting."""
        raw_cv = cnt.std() / cnt.mean()
        target = raw_cv * 0.001  # Fixed small CV ratio

        lo, hi = 0.0, 1.0
        for _ in range(300):  # Max iterations
            mid = (lo + hi) / 2
            w = self._weights_beta(cnt, mid)
            cv_mid = (w * cnt).std() / (w * cnt).mean()
            lo, hi = (mid, hi) if cv_mid > target else (lo, mid)

        return self._weights_beta(cnt, hi)

    def _weights_beta(self, cnt, beta):
        """Helper for computing beta-weighted degrees."""
        w = np.clip((cnt + self.eps) ** (-beta), self.w_min, self.w_max)
        return w / w.mean()
