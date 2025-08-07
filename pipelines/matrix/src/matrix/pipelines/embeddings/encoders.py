import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, TypeAlias, TypeVar

import numpy as np
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
    def version(self) -> str:
        """Provide version of the attribute encoder.

        The version of the attribute encoder is leveraged to ensure
        caching capabilities are able to properly invalidate the
        cache when the underlying version changes.

        For instance, the version can be the specific model used the LangChainEncoder,
        when the model changes, the cache is subsequently invalidated ensuring
        the new model is ran.
        """
        ...

    @abstractmethod
    async def apply(self, documents: Sequence[str]) -> list[list[float]]:
        """Encode text from documents into embeddings.

        Args:
            documents: list/tuple of strings that need an embedding

        Returns:
            A list of embeddings, one per document.
        """
        ...


class LangChainEncoder(AttributeEncoder):
    """Encoder class for OpenAI embeddings with efficient batch processing."""

    def __init__(
        self,
        encoder: OpenAIEmbeddings,
        dimensions: int,
        random_seed: Optional[int] = None,
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

    def version(self) -> str:
        return self._client.model

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    async def apply(self, documents: Sequence[str]) -> list[list[float]]:
        """Encode text from dataframe using OpenAI embeddings.

        Args:
            documents: sequence of strings for which you want their embeddings

        Returns:
            a list of embeddings, where each embedding is a list of floats
        """
        return await self._client.aembed_documents(list(documents))


class RandomizedEncoder(AttributeEncoder):
    """Encoder class for generating random embeddings."""

    def __init__(
        self,
        dimensions: int,
        random_seed: Optional[int] = None,
    ):
        """Initialize Randomized encoder.

        Args:
            dimensions: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
        """
        super().__init__(dimensions, random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

    def version(self) -> str:
        return "random"

    async def encode(self, documents: Sequence[str]) -> list[list[float]]:
        """Generate random embeddings for the input dataframe.

        Args:
            documents: list/tuple of strings that need an embedding

        Returns:
            A list of embeddings, one per document.
        """
        return [np.random.rand(self._embedding_dim).tolist() for _ in documents]


class PubmedBERTEncoder(AttributeEncoder):
    """Encoder class for PubmedBERT embeddings."""

    def __init__(
        self,
        dimensions: int,
        random_seed: Optional[int] = None,
        encoder: Optional[AttributeEncoder] = None,
        model_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ):
        """Initialize PubmedBERT encoder.

        Args:
            model_path: path to model, e.g.,
            dimensions: Dimension of the output embeddings
            random_seed: Random seed for reproducibility
            encoder: Encoder to use for embedding generation (dummy)
        """
        super().__init__(dimensions, random_seed)
        if random_seed is not None:
            np.random.seed(random_seed)

        self._model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def version(self) -> str:
        return self._model_path

    async def encode(self, documents: Sequence[str]) -> list[list[float]]:
        """Generate PubmedBERT embeddings for the input dataframe.

        Args:
            documents: list/tuple of strings that need an embedding

        Returns:
            A list of embeddings, one per document.
        """
        inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            return list(self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy())


class DummyResolver(AttributeEncoder):
    def __init__(self, **kwargs):
        super().__init__(0, 0)

    def version(self) -> str:
        return "dummy"

    async def apply(self, documents: Sequence[str]) -> list[list[float]]:
        return [[1.0, 2.0]] * len(documents)
