import numpy as np
import pandas as pd
from matrix_inject.inject import _extract_elements_in_list

from matrix.pipelines.modelling.model import ModelWrapper
from matrix.pipelines.modelling.nodes import apply_transformers


class ModelWithPreprocessing:
    """Lightweight wrapper that applies preprocessing before scoring."""

    def __init__(self, base_model: ModelWrapper, transformers_: dict, selected_features: list[str]) -> None:
        self._base_model = base_model
        self._transformers = transformers_
        self._features = selected_features

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = apply_transformers(data.copy(), self._transformers)
        feature_columns = _extract_elements_in_list(transformed.columns, self._features, True)
        return transformed[feature_columns]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(data)
        return self._base_model.predict_proba(transformed.values)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(data).argmax(axis=1)

    def __getattr__(self, item):
        return getattr(self._base_model, item)

    def __getstate__(self):
        """Custom serialization for pickle compatibility."""
        return {"base_model": self._base_model, "transformers": self._transformers, "features": self._features}

    def __setstate__(self, state):
        """Custom deserialization for pickle compatibility."""
        self._base_model = state["base_model"]
        self._transformers = state["transformers"]
        self._features = state["features"]
