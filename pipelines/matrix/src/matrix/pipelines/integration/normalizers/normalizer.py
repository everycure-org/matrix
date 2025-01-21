from abc import ABC, abstractmethod

import pandas as pd


class Normalizer(ABC):
    """Base class to represent normalizer strategies."""

    @abstractmethod
    async def apply(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to apply normalization."""
        ...
