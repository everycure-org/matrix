"""Custom Kedro dataset for FDA Drugs@FDA zipped JSON."""

import io
import json
import zipfile
from abc import ABC
from typing import Any

import pandas as pd
import requests
from kedro.io.core import AbstractDataset, DatasetError


class JSONZipDataset(AbstractDataset, ABC):
    """Load a JSON file embedded in a ZIP archive and return extracted records as a DataFrame.

    Example catalog entry:
        raw.fda_drug_list:
          type: core_entities.datasets.json_zip_dataset.JSONZipDataset
          url: https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip
          member_name: drug-drugsfda-0001-of-0001.json
                    records_path: results
    """

    def __init__(
        self,
        url: str,
        member_name: str,
        records_path: str | None = "results",
        request_timeout_seconds: int = 120,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._member_name = member_name
        self._records_path = records_path
        self._request_timeout_seconds = request_timeout_seconds
        self._metadata = metadata or {}

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        # NOTE: This method was partially generated using AI assistance.
        if self._records_path is None:
            extracted = payload
        else:
            extracted = payload
            for key in self._records_path.split("."):
                if not isinstance(extracted, dict):
                    raise DatasetError(
                        f"Expected dict while traversing records_path '{self._records_path}', got {type(extracted).__name__}"
                    )
                if key not in extracted:
                    raise DatasetError(f"Could not find key '{key}' in records_path '{self._records_path}'")
                extracted = extracted[key]

        if not isinstance(extracted, list):
            path_desc = self._records_path if self._records_path is not None else "<root>"
            raise DatasetError(f"Expected list at records_path '{path_desc}', got {type(extracted).__name__}")

        if not all(isinstance(item, dict) for item in extracted):
            raise DatasetError("Expected extracted records to be a list of JSON objects")

        return extracted

    def _load(self) -> pd.DataFrame:
        try:
            response = requests.get(self._url, timeout=self._request_timeout_seconds)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                with zf.open(self._member_name) as f:
                    payload = json.load(f)

            records = self._extract_records(payload)
            return pd.DataFrame(records)
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
            "records_path": self._records_path,
            "request_timeout_seconds": self._request_timeout_seconds,
        }
