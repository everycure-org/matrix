import asyncio
import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any

import aiohttp
import requests
import tenacity.stop
from jsonpath_ng import parse
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

from ...batch.pipeline import batched

logger = logging.getLogger(__name__)


class Normalizer(ABC):
    """Base class to represent normalizer strategies."""

    def __init__(
        self,
        conflate: bool,
        drug_chemical_conflate: bool,
        protocol_and_domain: str,
        get_normalized_nodes_path: str,
        description: bool = False,
        items_per_request: int = 1000,
    ) -> None:
        self._protocol_and_domain = protocol_and_domain
        self._get_normalized_nodes_path = get_normalized_nodes_path
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

        endpoint = f"{self._protocol_and_domain}{self._get_normalized_nodes_path}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url=endpoint, json=request_json) as resp:
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

    @functools.cache
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=tenacity.stop.stop_after_attempt(4),
        reraise=True,
        before_sleep=print,
    )
    def version(self) -> str:
        nn_openapi_json_url = f"https://{self._protocol_and_domain}/openapi.json"
        response = requests.get(nn_openapi_json_url, timeout=5)
        json_response = response.json()
        version = json_response["info"]["version"]
        return f"nodenorm-{self.get_source().lower()}-{version}"

    @abstractmethod
    def get_source(self) -> str: ...


class NCATSNodeNormalizer(Normalizer):
    """Class to represent normalizer from NCATS."""

    def __init__(
        self,
        conflate: bool,
        drug_chemical_conflate: bool,
        protocol_and_domain: str = "https://nodenorm.transltr.io",
        get_normalized_nodes_path: str = "/1.5/get_normalized_nodes",
        description: bool = False,
        items_per_request: int = 1000,
    ) -> None:
        super().__init__(
            conflate,
            drug_chemical_conflate,
            protocol_and_domain,
            get_normalized_nodes_path,
            description,
            items_per_request,
        )

    def get_source(self) -> str:
        return "NCATS"


class RENCINodeNormalizer(Normalizer):
    """Class to represent normalizer from RENCI."""

    def __init__(
        self,
        conflate: bool,
        drug_chemical_conflate: bool,
        protocol_and_domain: str = "https://nodenormalization-sri.renci.org",
        get_normalized_nodes_path: str = "/1.5/get_normalized_nodes",
        description: bool = False,
        items_per_request: int = 1000,
    ) -> None:
        super().__init__(
            conflate,
            drug_chemical_conflate,
            protocol_and_domain,
            get_normalized_nodes_path,
            description,
            items_per_request,
        )

    def get_source(self) -> str:
        return "RENCI"


class DummyNodeNormalizer(Normalizer):
    """Class to represent a dummy normalizer."""

    def __init__(
        self,
        conflate: bool,
        drug_chemical_conflate: bool,
        protocol_and_domain: str = "",
        get_normalized_nodes_path: str = "",
        description: bool = False,
        items_per_request: int = 1000,
    ) -> None:
        super().__init__(
            conflate,
            drug_chemical_conflate,
            protocol_and_domain,
            get_normalized_nodes_path,
            description,
            items_per_request,
        )

    @functools.cache
    def version(self) -> str:
        return f"nodenorm-{self.get_source().lower()}"

    def get_source(self) -> str:
        return "DUMMY"
