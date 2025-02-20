import asyncio
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeAlias, TypeVar

import numpy as np
import pandas as pd
import torch
from langchain_openai import OpenAIEmbeddings
from tenacity import Retrying, retry, stop_after_attempt, wait_exponential
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


async def embed_with_openai(
    documents: Iterable[str], init_workers: int, docs_per_async_task: int, model: str, request_timeout: int
) -> Generator[tuple[str, tuple[float]], None, None]:
    # OpenAIEmbeddings splits the list in batches, based on the class attribute chunk_size. To do so, it needs a collection instead of an iterable.
    tasks = []
    sem = asyncio.Semaphore(init_workers)
    embedder = OpenAIEmbeddings(model=model, request_timeout=request_timeout)
    async with asyncio.TaskGroup() as tg:
        for index, batch in enumerate(batched(documents, docs_per_async_task)):
            tasks.append(tg.create_task(call_for_minibatch(batch, embedder, index, sem)))
    print(f"final sem: {sem._value}")
    ret = itertools.chain.from_iterable(t.result() for t in tasks)
    return ret


def embed_with_openai_async(
    documents: Iterable[str],
    init_workers: int = 40,
    docs_per_async_task: int = 10_000,
    model: str = "text-embedding-3-small",
    request_timeout: int = 10,
) -> Iterator[tuple[str, tuple[float, ...]]]:
    res = asyncio.run(
        embed_with_openai(
            (_[0] for _ in documents),
            init_workers=init_workers,
            docs_per_async_task=docs_per_async_task,
            request_timeout=request_timeout,
            model=model,
        )
    )
    return res


async def call_for_minibatch(
    docs: Sequence[str], embedder: OpenAIEmbeddings, batch_index: int, sem: asyncio.Semaphore
) -> list[tuple[str, tuple[float, ...]]]:
    logger.warning(f"About to start actual coro on batch_index: {batch_index}, semaphore: {sem._value}")
    async with sem:
        logger.warning(f"Started coro on batch_index: {batch_index}, semaphore: {sem._value}")
        for index, attempt in enumerate(
            Retrying(wait=wait_exponential(multiplier=5, min=2, max=60), stop=stop_after_attempt(5))
        ):
            with attempt:
                tic = time.perf_counter()
                list_of_embeddings: list[list[float]] = await embedder.aembed_documents(docs)
                embedding_lengths = list(map(len, list_of_embeddings))
                embeddings = (tuple(embedding) for embedding in list_of_embeddings)
                notes = {
                    "batch_index": batch_index,
                    "attempt number": index,
                    "completion time (s)": time.perf_counter() - tic,
                    "number of documents": len(docs),
                    "number of embeddings": len(embedding_lengths),
                    "avg embedding length": sum(embedding_lengths) / len(embedding_lengths),
                }
                logger.warning(json.dumps(notes))
    if index == 0:
        logger.warning("increasing the semaphore")
        sem.release()  # increases the number of tasks that can run
    else:
        logger.warning(f"decreasing the semaphore from {batch_index=}")
        await sem.acquire()
        if sem.locked():
            logger.warning(f"Found we're locked in {batch_index=}")
            sem.release()  # ensure there's always 1
    return list(zip(docs, embeddings))


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


def lang_chain_encoder(
    texts: Iterable[str],
    dimensions: int,
    random_seed: Optional[int] = None,
    batch_size: int = 500,
    timeout: int = 10,
    openai_model: str = "text-embedding-3-small",
    openai_api_base: Optional[str] = None,
) -> Iterator[Tuple[str, List[float]]]:
    """Encode texts using OpenAI embeddings with efficient batch processing.

    Args:
        texts: Strings for which the embedding must be computed.
        dimensions: Output dimension of the output embeddings.
        random_seed: Random seed for reproducibility.
        batch_size: Batch size for efficient encoding.
        timeout: Timeout for OpenAI API requests.
        openai_model: Name of the model to use.
        openai_api_base: Endpoint for the OpenAI model.

    Returns:
        Tuples of the text and its embedding, for each text.
    """
    encoder = OpenAIEmbeddings(
        model=openai_model, request_timeout=timeout, openai_api_base=openai_api_base, dimensions=dimensions
    )

    def batched(iterable: Iterable[T], n: int, *, strict: bool = False) -> Iterator[Tuple[T]]:
        """Batch an iterable into tuples of size n."""
        if n < 1:
            raise ValueError("batch size must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch

    @retry(wait=wait_exponential(multiplier=10, min=2, max=180), stop=stop_after_attempt(5))
    def _encode(texts: List[str], encoder: OpenAIEmbeddings) -> List[List[float]]:
        """Encode a list of texts using the OpenAI encoder."""
        return encoder.embed_documents(texts=texts)

    for batch in batched(texts, batch_size):
        embeddings = _encode(list(batch), encoder)
        yield from zip(batch, embeddings)


class DummyResolver(AttributeEncoder):
    def __init__(self, **kwargs):
        super().__init__(0, 0)

    async def apply(self, documents: Sequence[str]) -> list[list[float]]:
        return [[1.0, 2.0]] * len(documents)
