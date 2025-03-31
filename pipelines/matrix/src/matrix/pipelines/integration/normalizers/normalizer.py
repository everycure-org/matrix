from abc import ABC, abstractmethod
from collections.abc import Collection


class Normalizer(ABC):
    """Base class to represent normalizer strategies."""

    @abstractmethod
    async def apply(self, strings: Collection[str], **kwargs) -> list[str | None]:
        """Function to apply normalization."""
        ...
