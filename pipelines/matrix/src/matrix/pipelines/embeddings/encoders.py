import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Optional, TypeAlias, TypeVar

import numpy as np
import pandas as pd
import torch
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

T: TypeAlias = TypeVar("T")


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
        dimensions: int,
        random_seed: Optional[int] = None,
        batch_size: int = 500,
    ):
        """Initialize OpenAI encoder.

        Args:
            dimensions: Output dimension of the output embeddings
            random_seed: Random seed for reproducibility
            batch_size: Batch size for efficient encoding
        """
        super().__init__(dimensions, random_seed)
        self.batch_size = batch_size

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def apply(self, texts: Iterable[str]) -> Iterator[tuple[str, list[float]]]:
        """Encode texts using OpenAI embeddings.

        Args:
            texts: strings for which the embedding must be computed

        Returns:
            tuples of the text and its embedding, for each text

        """
        # TODO: tune timeout
        encoder = OpenAIEmbeddings(model="text-embedding-3-small", timeout=10)
        pid = os.getpid()
        for index, batch in enumerate(self.batched(texts, self.batch_size)):
            # Note: on error, due to the retry logic and iterables, we might lose a batch. Should we remove the retry logic?
            # Write a test for this case, that when embed_documents errors out, it still returns for all texts an embedding.
            logger.debug('{"PID": %d, "batch": %d}', pid, index)
            embeddings = encoder.embed_documents(texts=batch)
            yield from zip(batch, embeddings)

    @staticmethod
    def batched(iterable: Iterable[T], n: int, *, strict: bool = False) -> Iterator[tuple[T]]:
        # Taken from the recipe at https://docs.python.org/3/library/itertools.html#itertools.batched , which is available by default in 3.12
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("batch size must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch


class RandomizedEncoder(AttributeEncoder):
    """Encoder class for generating random embeddings."""

    def __init__(self, dimensions: int, random_seed: Optional[int] = None):
        """Initialize Randomized encoder.

        Args:
            dimensions: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
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
