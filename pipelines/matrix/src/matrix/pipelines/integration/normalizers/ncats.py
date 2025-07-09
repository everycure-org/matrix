import asyncio
import logging
from collections.abc import Collection
from typing import Any

import aiohttp
from jsonpath_ng import parse
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

from ...batch.pipeline import batched
from .normalizer import Normalizer

logger = logging.getLogger(__name__)


class NCATSNodeNormalizer(Normalizer):
    """Class to represent normalizer from translator."""

    def __init__(
        self,
        endpoint: str,
        conflate: bool,
        drug_chemical_conflate: bool,
        description: bool = False,
        items_per_request: int = 1000,
    ) -> None:
        self._endpoint = endpoint
        self._conflate = conflate
        self._drug_chemical_conflate = drug_chemical_conflate
        self._description = description
        self._json_parser = {
            "id": parse("$.id.identifier"),
            "category": parse("$.type.[*]"),  # category list from NN
        }
        self._items_per_request = items_per_request

    async def apply(self, strings: Collection[str], **kwargs) -> list[str | None]:
        results = []
        # The node normalizer doesn't deal well with large batch sizes, so we'll do the chunking on our side.
        for batch in batched(strings, self._items_per_request):
            results.extend(await self.normalize_batch(batch))
        return results

    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=print,
    )
    async def normalize_batch(self, batch: Collection[str]):
        request_json = {
            "curies": tuple(batch),  # must be json serializable
            "conflate": self._conflate,
            "drug_chemical_conflate": self._drug_chemical_conflate,
            "description": self._description,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, json=request_json) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    logger.debug(response_json)
                else:
                    logger.warning(f"Node norm response code: {resp.status}")
                    resp_text = await resp.text()
                    logger.error(resp_text)  # To extract more info from the error

                resp.raise_for_status()

        return [self._extract(curie, response_json, self._json_parser) for curie in batch]

    @staticmethod
    def _extract(
        id: str, response: dict[str, Any], json_parser: parse, default_normalizer_category: str = "biolink:NamedThing"
    ) -> dict[str, Any] | None:
        """Extract normalized IDs from the response using the json parser."""
        try:
            curie_info = response.get(id)
            if curie_info is None:
                return {"normalized_id": None, "normalized_categories": [default_normalizer_category]}
            normalized_id = str(json_parser["id"].find(curie_info)[0].value)
            # Find categories for CURIE in NN. If no categories are present, use default
            categories = [match.value for match in json_parser["category"].find(curie_info)] or [
                default_normalizer_category
            ]

            return {"normalized_id": normalized_id, "normalized_categories": categories}
        except (IndexError, KeyError):
            logger.debug(f"Not able to normalize for {id}: {response.get(id)}, {json_parser}")
            return {"normalized_id": None, "normalized_categories": [default_normalizer_category]}
