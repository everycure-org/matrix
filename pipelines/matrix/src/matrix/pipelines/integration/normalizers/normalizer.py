from abc import ABC, abstractmethod

import pandas as pd


class Normalizer(ABC):
    """Base class to represent normalizer strategies."""

    @abstractmethod
    async def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to apply normalization."""
        ...


class NopNormalizer(ABC):
    """Normalizer that does not perform normalization."""

    async def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to apply normalization."""

        df["normalized_id"] = df["id"]
        df["normalized_id"] = df["normalized_id"].astype(pd.StringDtype())

        return df
