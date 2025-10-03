import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _dataframe_to_mapping_dict(mapping_df: pd.DataFrame) -> Dict[str, str]:
    """Convert DataFrame with subject_id/object_id columns to a mapping dictionary."""
    return {row["subject_id"]: row["object_id"] for _, row in mapping_df.iterrows()}


def _extract_records(source_data: Any, config: Dict[str, Any]) -> Any:
    """Extract records from source data, handling nested structures if data_path is configured."""
    return source_data[config["data_path"]] if "data_path" in config else source_data


def _add_parsed_record(
    result: Dict[str, Dict[str, Any]], pks_id: str, record: Dict[str, Any], id_value: Any, config: Dict[str, Any]
) -> None:
    """Build extracted data and add to result dictionary, initializing nested dict if needed."""
    if pks_id not in result:
        result[pks_id] = {}

    result[pks_id][config["name"]] = {
        config["id_column"]: id_value,
        **{field: record[field] for field in config["extracted_metadata"] if field in record},
    }


def parse_infores(source_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse infores catalog - YAML format with data_path structure.

    Args:
        source_data: YAML dict containing information resources
        config: Configuration for this source from parameters.yml

    Returns:
        Dict keyed by PKS ID with extracted metadata
    """
    records = _extract_records(source_data, config)
    result = {}

    for record in records:
        pks_id = str(record.get(config["id_column"], "")).replace("infores:", "")
        if not pks_id:
            continue

        _add_parsed_record(result, pks_id, record, record.get(config["id_column"], ""), config)

    return result


def parse_reusabledata(source_data: Any, mapping_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse reusabledata.org - JSON list format that requires ID mapping.

    Args:
        source_data: JSON list of reusabledata records
        mapping_df: DataFrame mapping reusabledata IDs to infores IDs
        config: Configuration for this source from parameters.yml

    Returns:
        Dict keyed by PKS ID with extracted metadata
    """
    mapping_dict = _dataframe_to_mapping_dict(mapping_df)
    result = {}

    for record in source_data:
        original_id = record.get(config["original_id_column"], "")
        if not original_id:
            continue

        mapped_id = mapping_dict.get(original_id, original_id)
        pks_id = str(mapped_id).replace("infores:", "")

        _add_parsed_record(result, pks_id, record, mapped_id, config)

    return result


def parse_kgregistry(
    source_data: Dict[str, Any], mapping_df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Parse KG registry - YAML format with special infores_id handling.

    Args:
        source_data: YAML dict containing KG registry data
        mapping_df: DataFrame mapping kgregistry IDs to infores IDs
        config: Configuration for this source from parameters.yml

    Returns:
        Dict keyed by PKS ID with extracted metadata
    """
    mapping_dict = _dataframe_to_mapping_dict(mapping_df)
    records = _extract_records(source_data, config)
    result = {}

    for record in records:
        original_id = record.get(config["original_id_column"], "")
        if not original_id:
            continue

        # Special handling: prefer infores_id if present
        mapped_id = record["infores_id"] if record.get("infores_id") else mapping_dict.get(original_id, original_id)
        pks_id = str(mapped_id).replace("infores:", "")

        _add_parsed_record(result, pks_id, record, mapped_id, config)

        # Add updated_id field for kgregistry special case
        if "infores_id" in record:
            result[pks_id][config["name"]]["updated_id"] = record["infores_id"]

    return result


def parse_matrix_dataframe(source_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse Matrix pandas DataFrame sources (curated metadata or reviews).

    Args:
        source_data: pandas DataFrame with Matrix data
        config: Configuration for this source from parameters.yml

    Returns:
        Dict keyed by PKS ID with extracted metadata
    """
    result = {}

    for record in source_data.to_dict("records"):
        pks_id = str(record.get(config["id_column"], "")).replace("infores:", "")
        if not pks_id:
            continue

        _add_parsed_record(result, pks_id, record, record.get(config["id_column"], ""), config)

    return result
