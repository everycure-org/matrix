"""Custom Kedro dataset for FDA Drugs@FDA zipped JSON."""

import io
import json
import zipfile
from abc import ABC
from typing import Any

import requests
from kedro.io.core import AbstractDataset, DatasetError


class JSONZipDataset(AbstractDataset, ABC):
    """Load a JSON file embedded in a remote ZIP archive.

    Example catalog entry:
        raw.fda_drug_list:
          type: core_entities.datasets.fda.JSONZipDataset
          url: https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip
          member_name: drug-drugsfda-0001-of-0001.json
    """

    def __init__(
        self,
        url: str,
        member_name: str,
        request_timeout_seconds: int = 120,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._member_name = member_name
        self._request_timeout_seconds = request_timeout_seconds
        self._metadata = metadata or {}

    def _load(self) -> dict[str, Any]:
        try:
            response = requests.get(self._url, timeout=self._request_timeout_seconds)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                with zf.open(self._member_name) as f:
                    payload = json.load(f)
            return payload
        except Exception as exc:
            raise DatasetError(
                f"Failed to load JSON member '{self._member_name}' from ZIP URL '{self._url}': {exc}"
            ) from exc

    def _save(self, data: dict[str, Any]) -> None:
        raise DatasetError("JSONZipDataset is read-only and does not support save().")

    def _describe(self) -> dict[str, Any]:
        return {
            "url": self._url,
            "member_name": self._member_name,
            "request_timeout_seconds": self._request_timeout_seconds,
        }
