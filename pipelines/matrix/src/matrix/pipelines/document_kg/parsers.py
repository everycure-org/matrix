import logging
from typing import Any, Dict, Optional

import pandas as pd

from .parser import ExternalRegistryParser, MatrixCuratedParser

logger = logging.getLogger(__name__)


def _dataframe_to_mapping_dict(mapping_df: pd.DataFrame) -> Dict[str, str]:
    """Convert SSSOM mapping DataFrame to dictionary."""
    return {row["subject_id"]: row["object_id"] for _, row in mapping_df.iterrows()}


def _add_parsed_record(
    result: Dict[str, Dict[str, Any]],
    pks_id: str,
    source_name: str,
    id_value: Any,
    record: Dict[str, Any],
    extracted_fields: list[str],
    id_column: str,
) -> None:
    if pks_id not in result:
        result[pks_id] = {}

    result[pks_id][source_name] = {
        id_column: id_value,
        **{field: record[field] for field in extracted_fields if field in record},
    }


class InforesParser(ExternalRegistryParser):
    """Parser for Information Resource Registry (biolink/infores).

    Parses the infores catalog YAML to extract PKS metadata.
    No ID mapping required as infores IDs are canonical.
    """

    def parse(
        self, source_data: Dict[str, Any], mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        data_path = self.config.get("data_path")
        records = source_data[data_path] if data_path else source_data

        result = {}

        for record in records:
            pks_id = str(record.get(self.id_column, "")).replace("infores:", "")
            if not pks_id:
                continue

            _add_parsed_record(
                result=result,
                pks_id=pks_id,
                source_name=self.name,
                id_value=record.get(self.id_column, ""),
                record=record,
                extracted_fields=self.extracted_metadata,
                id_column=self.id_column,
            )

        logger.info(f"Parsed {len(result)} PKS from {self.name}")
        return result


class ReusableDataParser(ExternalRegistryParser):
    """Parser for reusabledata.org registry."""

    def __init__(self, *args, original_id_column: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_id_column = original_id_column

    def parse(
        self, source_data: Any, mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        if mapping_data is None:
            logger.warning(
                f"{self.name} parser requires mapping_data for ID translation.IDs may not map correctly to infores."
            )
            mapping_dict = {}
        else:
            mapping_dict = _dataframe_to_mapping_dict(mapping_data)

        result = {}

        for record in source_data:
            original_id = record.get(self.original_id_column, "")
            if not original_id:
                continue

            mapped_id = mapping_dict.get(original_id, original_id)
            pks_id = str(mapped_id).replace("infores:", "")

            _add_parsed_record(
                result=result,
                pks_id=pks_id,
                source_name=self.name,
                id_value=mapped_id,
                record=record,
                extracted_fields=self.extracted_metadata,
                id_column=self.id_column,
            )

        logger.info(f"Parsed {len(result)} PKS from {self.name}")
        return result


class KGRegistryParser(ExternalRegistryParser):
    """Parser for Knowledge Graph Hub Registry.

    Parses KG Registry YAML with optional SSSOM mapping.
    Prefers infores_id field in record if present, otherwise uses mapping.
    """

    def __init__(self, *args, original_id_column: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_id_column = original_id_column

    def parse(
        self, source_data: Dict[str, Any], mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        if mapping_data is None:
            logger.warning(f"{self.name} parser works best with mapping_data")
            mapping_dict = {}
        else:
            mapping_dict = _dataframe_to_mapping_dict(mapping_data)

        data_path = self.config.get("data_path")
        records = source_data[data_path] if data_path else source_data

        result = {}

        for record in records:
            original_id = record.get(self.original_id_column, "")
            if not original_id:
                continue

            mapped_id = record["infores_id"] if record.get("infores_id") else mapping_dict.get(original_id, original_id)
            pks_id = str(mapped_id).replace("infores:", "")

            _add_parsed_record(
                result=result,
                pks_id=pks_id,
                source_name=self.name,
                id_value=mapped_id,
                record=record,
                extracted_fields=self.extracted_metadata,
                id_column=self.id_column,
            )

            if "infores_id" in record:
                result[pks_id][self.name]["updated_id"] = record["infores_id"]

        logger.info(f"Parsed {len(result)} PKS from {self.name}")
        return result


class MatrixCuratedMetadataParser(MatrixCuratedParser):
    """Parser for Matrix curated PKS metadata (license information)."""

    def parse(
        self, source_data: pd.DataFrame, mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        for record in source_data.to_dict("records"):
            pks_id = str(record.get(self.id_column, "")).replace("infores:", "")
            if not pks_id:
                continue

            _add_parsed_record(
                result=result,
                pks_id=pks_id,
                source_name=self.name,
                id_value=record.get(self.id_column, ""),
                record=record,
                extracted_fields=self.extracted_metadata,
                id_column=self.id_column,
            )

        logger.info(f"Parsed {len(result)} PKS from {self.name}")
        return result


class MatrixReviewsParser(MatrixCuratedParser):
    """Parser for Matrix PKS reviews (drug repurposing relevancy scores)."""

    def parse(
        self, source_data: pd.DataFrame, mapping_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        for record in source_data.to_dict("records"):
            pks_id = str(record.get(self.id_column, "")).replace("infores:", "")
            if not pks_id:
                continue

            _add_parsed_record(
                result=result,
                pks_id=pks_id,
                source_name=self.name,
                id_value=record.get(self.id_column, ""),
                record=record,
                extracted_fields=self.extracted_metadata,
                id_column=self.id_column,
            )

        logger.info(f"Parsed {len(result)} PKS from {self.name}")
        return result
