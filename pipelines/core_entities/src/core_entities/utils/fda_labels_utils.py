"""
FDA OTC monograph checker for openFDA label data.
All API/query behaviour is parameter-driven, so this utility can be reused by future pipelines.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_FDA_API_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_API_KEY_ENV_VAR = "FDA_API_KEY"
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = (1, 3, 7)
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_RESULT_LIMIT = 1
DEFAULT_MONOGRAPH_APPLICATION_PREFIX = "M"
DEFAULT_SUBSTANCE_NAME_FIELD = "openfda.substance_name"
DEFAULT_APPLICATION_NUMBER_FIELD = "openfda.application_number"
DEFAULT_BOOLEAN_OPERATOR = "AND"


@dataclass(frozen=True)
class FdaLabelsQueryConfig:
    substance_name_field: str = DEFAULT_SUBSTANCE_NAME_FIELD
    application_number_field: str = DEFAULT_APPLICATION_NUMBER_FIELD
    application_number_prefix: str = DEFAULT_MONOGRAPH_APPLICATION_PREFIX
    boolean_operator: str = DEFAULT_BOOLEAN_OPERATOR
    additional_clauses: tuple[str, ...] = ()
    uppercase_substance_name: bool = True

    @classmethod
    def from_dict(cls, config: Mapping[str, object] | None) -> "FdaLabelsQueryConfig":
        if not isinstance(config, Mapping):
            return cls()

        additional_clauses = config.get("additional_clauses", [])
        clauses: tuple[str, ...] = ()
        if isinstance(additional_clauses, list):
            clauses = tuple(
                clause.strip() for clause in additional_clauses if isinstance(clause, str) and clause.strip()
            )

        boolean_operator = (
            str(config.get("boolean_operator", DEFAULT_BOOLEAN_OPERATOR) or DEFAULT_BOOLEAN_OPERATOR).strip().upper()
        )
        if not boolean_operator:
            boolean_operator = DEFAULT_BOOLEAN_OPERATOR

        return cls(
            substance_name_field=str(
                config.get("substance_name_field", DEFAULT_SUBSTANCE_NAME_FIELD) or DEFAULT_SUBSTANCE_NAME_FIELD
            ).strip(),
            application_number_field=str(
                config.get("application_number_field", DEFAULT_APPLICATION_NUMBER_FIELD)
                or DEFAULT_APPLICATION_NUMBER_FIELD
            ).strip(),
            application_number_prefix=str(
                config.get("application_number_prefix", DEFAULT_MONOGRAPH_APPLICATION_PREFIX)
                or DEFAULT_MONOGRAPH_APPLICATION_PREFIX
            ).strip(),
            boolean_operator=boolean_operator,
            additional_clauses=clauses,
            uppercase_substance_name=bool(config.get("uppercase_substance_name", True)),
        )


@dataclass(frozen=True)
class FdaLabelsConfig:
    api_url: str = DEFAULT_FDA_API_URL
    api_key_env_var: str = DEFAULT_API_KEY_ENV_VAR
    api_key: str | None = None
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_backoff_seconds: tuple[int, ...] = DEFAULT_RETRY_BACKOFF_SECONDS
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS
    result_limit: int = DEFAULT_RESULT_LIMIT
    monograph_application_prefix: str = DEFAULT_MONOGRAPH_APPLICATION_PREFIX
    query: FdaLabelsQueryConfig = field(default_factory=FdaLabelsQueryConfig)

    @classmethod
    def from_dict(cls, config: Mapping[str, object] | None) -> "FdaLabelsConfig":
        if not isinstance(config, Mapping):
            return cls()

        api_config = config.get("api", {})
        query_config = config.get("query", {})
        if not isinstance(api_config, Mapping):
            api_config = {}
        if not isinstance(query_config, Mapping):
            query_config = {}

        return cls(
            api_url=str(api_config.get("api_url", DEFAULT_FDA_API_URL) or DEFAULT_FDA_API_URL).strip(),
            api_key_env_var=str(
                api_config.get("api_key_env_var", DEFAULT_API_KEY_ENV_VAR) or DEFAULT_API_KEY_ENV_VAR
            ).strip(),
            api_key=_non_empty_string_or_none(api_config.get("api_key")),
            max_concurrent=_coerce_positive_int(api_config.get("max_concurrent"), DEFAULT_MAX_CONCURRENT),
            retry_attempts=_coerce_positive_int(api_config.get("retry_attempts"), DEFAULT_RETRY_ATTEMPTS),
            retry_backoff_seconds=_coerce_retry_backoff(
                api_config.get("retry_backoff_seconds"), DEFAULT_RETRY_BACKOFF_SECONDS
            ),
            request_timeout_seconds=_coerce_positive_int(
                api_config.get("request_timeout_seconds"), DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
            result_limit=_coerce_positive_int(api_config.get("result_limit"), DEFAULT_RESULT_LIMIT),
            monograph_application_prefix=str(
                config.get("monograph_application_prefix", DEFAULT_MONOGRAPH_APPLICATION_PREFIX)
                or DEFAULT_MONOGRAPH_APPLICATION_PREFIX
            ).strip(),
            query=FdaLabelsQueryConfig.from_dict(query_config),
        )


@dataclass
class DrugResult:
    drug_name: str
    status: str  # "OTC_MONOGRAPH" | "NOT_MONOGRAPH" | "NO_RESULTS" | "ERROR"
    application_numbers: list[str] = field(default_factory=list)
    total_matches: int = 0
    error_msg: str = ""


def load_fda_labels_config(parameters: Mapping[str, object] | None = None) -> FdaLabelsConfig:
    """Create an `FdaLabelsConfig` from a `params:*` dictionary."""
    return FdaLabelsConfig.from_dict(parameters)


def build_search_query(drug_name: str, query_config: FdaLabelsQueryConfig) -> str:
    normalized_name = " ".join(drug_name.split())
    if query_config.uppercase_substance_name:
        normalized_name = normalized_name.upper()

    clauses: list[str] = []
    if query_config.substance_name_field:
        clauses.append(f'{query_config.substance_name_field}:"{normalized_name}"')

    if query_config.application_number_field and query_config.application_number_prefix:
        clauses.append(f"{query_config.application_number_field}:{query_config.application_number_prefix}*")

    clauses.extend(query_config.additional_clauses)
    if not clauses:
        raise ValueError("At least one query clause must be configured for the FDA label search.")

    return f" {query_config.boolean_operator} ".join(clauses)


def build_url(drug_name: str, config: FdaLabelsConfig | Mapping[str, object] | None = None) -> str:
    effective_config = _as_fda_labels_config(config)
    params = {
        "search": build_search_query(drug_name, effective_config.query),
        "limit": effective_config.result_limit,
    }
    api_key = _resolve_api_key(effective_config)
    if api_key:
        params["api_key"] = api_key
    return f"{effective_config.api_url}?{urlencode(params)}"


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    drug_name: str,
    semaphore: asyncio.Semaphore,
    config: FdaLabelsConfig | Mapping[str, object] | None = None,
) -> DrugResult:
    effective_config = _as_fda_labels_config(config)
    timeout = aiohttp.ClientTimeout(total=effective_config.request_timeout_seconds)
    request_params: dict[str, str | int] = {
        "search": build_search_query(drug_name, effective_config.query),
        "limit": effective_config.result_limit,
    }
    api_key = _resolve_api_key(effective_config)
    if api_key:
        request_params["api_key"] = api_key

    async with semaphore:
        for attempt in range(1, effective_config.retry_attempts + 1):
            wait_seconds = _wait_time_for_attempt(
                effective_config.retry_backoff_seconds,
                attempt_index=attempt - 1,
            )
            try:
                async with session.get(
                    effective_config.api_url,
                    params=request_params,
                    timeout=timeout,
                ) as response:
                    if response.status == 429:
                        retry_after = _coerce_non_negative_int(
                            response.headers.get("Retry-After"),
                            wait_seconds,
                        )
                        logger.warning("Rate limited for %s. Retrying in %ss.", drug_name, retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status == 404:
                        return DrugResult(drug_name=drug_name, status="NO_RESULTS")

                    response.raise_for_status()
                    data = await response.json(content_type=None)

                    total_matches = _extract_total_matches(data)
                    if total_matches == 0:
                        return DrugResult(drug_name=drug_name, status="NO_RESULTS")

                    app_numbers = _extract_application_numbers(data)
                    has_monograph = any(
                        number.upper().startswith(effective_config.monograph_application_prefix.upper())
                        for number in app_numbers
                    )
                    return DrugResult(
                        drug_name=drug_name,
                        status="OTC_MONOGRAPH" if has_monograph else "NOT_MONOGRAPH",
                        application_numbers=app_numbers,
                        total_matches=total_matches,
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == effective_config.retry_attempts:
                    return DrugResult(drug_name=drug_name, status="ERROR", error_msg=str(exc))
                logger.warning(
                    "Retry %s/%s for %s due to error: %s",
                    attempt,
                    effective_config.retry_attempts,
                    drug_name,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

    return DrugResult(drug_name=drug_name, status="ERROR", error_msg="Exhausted retries")


async def run(
    drugs: list[str],
    config: FdaLabelsConfig | Mapping[str, object] | None = None,
) -> list[DrugResult]:
    effective_config = _as_fda_labels_config(config)
    if not drugs:
        return []

    semaphore = asyncio.Semaphore(effective_config.max_concurrent)
    connector = aiohttp.TCPConnector(limit=effective_config.max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_with_retry(session, drug, semaphore, effective_config) for drug in drugs]

        results: list[DrugResult] = []
        completed = 0
        total = len(tasks)
        start_time = time.monotonic()

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            elapsed_seconds = time.monotonic() - start_time
            request_rate = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
            eta_seconds = ((total - completed) / request_rate) if request_rate > 0 else 0.0
            logger.info(
                "[%s/%s] %s -> %s (%.1f req/s, ETA %.0fs)",
                completed,
                total,
                result.drug_name,
                result.status,
                request_rate,
                eta_seconds,
            )

    return results


def run_sync(
    drugs: list[str],
    config: FdaLabelsConfig | Mapping[str, object] | None = None,
) -> list[DrugResult]:
    """Synchronous wrapper for Kedro nodes and scripts."""
    return asyncio.run(run(drugs, config=config))


def _resolve_api_key(config: FdaLabelsConfig) -> str | None:
    if config.api_key:
        return config.api_key
    return os.getenv(config.api_key_env_var)


def _as_fda_labels_config(config: FdaLabelsConfig | Mapping[str, object] | None) -> FdaLabelsConfig:
    if isinstance(config, FdaLabelsConfig):
        return config
    if isinstance(config, Mapping):
        return FdaLabelsConfig.from_dict(config)
    return FdaLabelsConfig()


def _non_empty_string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_non_negative_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _coerce_retry_backoff(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list):
        return default

    parsed = [_coerce_non_negative_int(item, -1) for item in value]
    filtered = [item for item in parsed if item >= 0]
    return tuple(filtered) if filtered else default


def _wait_time_for_attempt(retry_backoff_seconds: tuple[int, ...], attempt_index: int) -> int:
    if len(retry_backoff_seconds) == 0:
        return 0
    safe_index = min(attempt_index, len(retry_backoff_seconds) - 1)
    return retry_backoff_seconds[safe_index]


def _extract_total_matches(data: object) -> int:
    if not isinstance(data, dict):
        return 0
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        return 0
    results = meta.get("results", {})
    if not isinstance(results, dict):
        return 0
    return _coerce_non_negative_int(results.get("total"), 0)


def _extract_application_numbers(data: object) -> list[str]:
    if not isinstance(data, dict):
        return []

    results = data.get("results", [])
    if not isinstance(results, list) or len(results) == 0:
        return []

    first_result = results[0]
    if not isinstance(first_result, dict):
        return []

    openfda = first_result.get("openfda", {})
    if not isinstance(openfda, dict):
        return []

    application_numbers = openfda.get("application_number", [])
    if isinstance(application_numbers, str):
        value = application_numbers.strip()
        return [value] if value else []

    if not isinstance(application_numbers, list):
        return []

    normalized = []
    for value in application_numbers:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                normalized.append(stripped)
    return normalized
