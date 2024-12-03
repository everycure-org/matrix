from abc import ABC, abstractmethod
from typing import Dict, Any
from jsonpath_ng import parse

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

import asyncio
import aiohttp
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Normalizer(ABC):
    """Base class to represent normalizer strategies."""

    @abstractmethod
    async def apply(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Function to apply normalization."""
        ...


class NCATSNodeNormalizer(Normalizer):
    """Class to represent nornalizer from translator."""

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
                    ids = self._extract_ids(response_json, self._json_parser)
                else:
                    logger.warning(f"Node norm response code: {resp.status}")
                    resp_text = await resp.text()
                    logger.debug(resp_text)

                resp.raise_for_status()

        df["normalized_id"] = ids
        return df

    @staticmethod
    def _extract_ids(response: Dict[str, Any], json_parser: parse):
        ids = {}
        for key, item in response.items():
            logger.debug(f"Response for key {key}: {response.get(key)}")  # Log the response for each key
            try:
                ids[key] = json_parser.find(item)[0].value
            except (IndexError, KeyError):
                logger.debug(f"Not able to normalize for {key}: {item}, {json_parser}")
                ids[key] = None

        return ids
