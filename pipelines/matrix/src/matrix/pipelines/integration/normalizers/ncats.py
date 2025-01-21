import asyncio
import logging
from abc import ABC
from typing import Any, Dict

import aiohttp
import pandas as pd
from jsonpath_ng import parse
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

from .normalizer import Normalizer

logger = logging.getLogger(__name__)


class NCATSNodeNormalizer(Normalizer):
    """Class to represent normalizer from translator."""

    def __init__(self, endpoint: str, conflate: bool, drug_chemical_conflate: bool, description: bool = False) -> None:
        self._endpoint = endpoint
        self._conflate = conflate
        self._drug_chemical_conflate = drug_chemical_conflate
        self._description = description
        self._json_parser = parse("$.id.identifier")  # FUTURE: Ensure we can update

    @retry(
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=print,
    )
    async def apply(self, df: pd.DataFrame, **kwargs):
        curies = df["id"].tolist()
        request_json = {
            "curies": curies,
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
                    logger.debug(resp_text)

                resp.raise_for_status()

        df["normalized_id"] = [self._extract_id(curie, response_json, self._json_parser) for curie in curies]
        df["normalized_id"] = df["normalized_id"].astype(pd.StringDtype())
        return df

    @staticmethod
    def _extract_id(id: str, response: Dict[str, Any], json_parser: parse) -> Dict[str, Any]:
        """Extract normalized IDs from the response using the json parser."""
        try:
            return json_parser.find(response.get(id))[0].value
        except (IndexError, KeyError):
            logger.debug(f"Not able to normalize for {id}: {response.get(id)}, {json_parser}")
            return None
