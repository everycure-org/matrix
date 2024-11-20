from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
# NOTE: This file was partially generated using AI assistance.


class AttributeEncoder(ABC):
    """Base class for encoders that convert text into embeddings."""

    def __init__(self, output_dim: int = 512, random_seed: Optional[int] = None):
        """Initialize base encoder.

        Args:
            output_dim: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
        """
        self._embedding_dim = output_dim
        self._random_seed = random_seed

    @abstractmethod
    async def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode text from dataframe into embeddings.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        pass


class OpenAIEncoder(AttributeEncoder):
    """Encoder class for OpenAI embeddings with efficient batch processing."""

    def __init__(self, encoder: str, dimensions: int = 512, random_seed: Optional[int] = None, timeout: int = 10):
        """Initialize OpenAI encoder.

        Args:
            key: OpenAI API key
            encoder: Name of the embedding model
            output_dim: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
            timeout: Timeout for OpenAI API requests
        """
        super().__init__(dimensions, random_seed)
        self._client = encoder

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode text from dataframe using OpenAI embeddings.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        try:
            combined_texts = df["text_to_embed"].tolist()
            df = df.copy()
            df["embedding"] = await self._client.aembed_documents(combined_texts)
            df = df.drop(columns=["text_to_embed"])
            return df
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise e
