from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModel, AutoTokenizer


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
    async def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Encode text from dataframe into embeddings.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        ...


class LangChainEncoder(AttributeEncoder):
    """Encoder class for OpenAI embeddings with efficient batch processing."""

    def __init__(
        self,
        encoder: OpenAIEmbeddings,
        dimensions: int,
        random_seed: Optional[int] = None,
        timeout: int = 10,
    ):
        """Initialize OpenAI encoder.

        Args:
            encoder: Name of the embedding model
            output_dim: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
            timeout: Timeout for OpenAI API requests
        """
        super().__init__(dimensions, random_seed)
        self._client = encoder

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def apply(self, df: pd.DataFrame, input_features: List[str], max_input_len: int) -> pd.DataFrame:
        """Encode text from dataframe using OpenAI embeddings.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        try:
            df["text_to_embed"] = df[input_features].apply(lambda row: "".join(row)[0:max_input_len], axis=1)
            combined_texts = df["text_to_embed"].tolist()
            df["embedding"] = await self._client.aembed_documents(combined_texts)
            df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
            df = df.drop(columns=["text_to_embed", *input_features])
            return df
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise e


class RandomizedEncoder(AttributeEncoder):
    """Encoder class for generating random embeddings."""

    def __init__(self, dimensions: int, random_seed: Optional[int] = None, encoder: Optional[AttributeEncoder] = None):
        """Initialize Randomized encoder.

        Args:
            dimensions: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
            encoder: Encoder to use for embedding generation (dummy)
        """
        super().__init__(dimensions, random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

    async def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate random embeddings for the input dataframe.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        df = df.copy()
        # Generate random embeddings
        df["embedding"] = [np.random.rand(self._embedding_dim).astype(np.float32) for _ in range(len(df))]
        df = df.drop(columns=["text_to_embed"])
        return df


class PubmedBERTEncoder(AttributeEncoder):
    """Encoder class for PubmedBERT embeddings."""

    def __init__(self, dimensions: int, random_seed: Optional[int] = None, encoder: Optional[AttributeEncoder] = None):
        """Initialize PubmedBERT encoder.

        Args:
            dimensions: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
            encoder: Encoder to use for embedding generation (dummy)
        """
        super().__init__(dimensions, random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    async def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate PubmedBERT embeddings for the input dataframe.

        Args:
            df: Input dataframe containing 'text_to_embed' column

        Returns:
            DataFrame with new 'embedding' column and 'text_to_embed' removed
        """
        df = df.copy()
        feat_list = df["text_to_embed"].tolist()
        inputs = self.tokenizer(
            feat_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        df["embedding"] = list(embeddings)
        df = df.drop(columns=["text_to_embed"])
        return df
