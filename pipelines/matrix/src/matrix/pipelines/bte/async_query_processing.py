"""This module handles asynchronous queries processing for the BTE pipeline."""

import asyncio
import logging
import httpx
import json
from .query_generation import remove_empty_fields
from .bte_pydantic import InitialResponse, StatusResponse, FinalResponse, Query
from typing import List, Dict, Optional, Any
from pydantic import ValidationError

MAX_CONCURRENT_REQUESTS = 16
TOTAL_TIMEOUT = 280.0
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 250.0
ASYNC_QUERY_URL = "https://bte.hueb.org/v1/asyncquery"
RETRY_ATTEMPTS = 8
MAX_RETRIES = 4
RETRY_DELAY = 4
HTTP_TIMEOUT_LIMIT = 280.0

timeout_config = httpx.Timeout(
    timeout=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT
)


async def retry_request(
    request_func: callable,
    *args,
    retries: int = 3,
    delay: int = 2,
    headers=None,
    **kwargs,
) -> httpx.Response:
    """Retry an HTTP request with exponential backoff.

    :param request_func: The HTTP request function to call.
    :param retries: Number of retries before giving up.
    :param delay: Initial delay between retries, will be doubled after each retry.
    :param headers: Optional headers to include in the request.
    :return: The HTTP response.
    :raises: Raises an exception if all retries fail.
    """
    if headers is None:
        headers = {}
    headers["Content-Type"] = "application/json"

    for attempt in range(retries):
        try:
            response = await request_func(*args, headers=headers, **kwargs)
            logging.debug(
                f"Response received: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logging.error(f"Error: {e}. Attempt {attempt + 1} of {retries}")
        if attempt < retries - 1:
            await asyncio.sleep(delay * (2**attempt))
        else:
            raise


async def post_async_query(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """Post an asynchronous query to the specified URL.

    :param client: HTTPX async client instance.
    :param url: URL to which the query is posted.
    :param payload: Payload of the query.
    :return: JSON dictionary response.
    """
    cleaned_payload = remove_empty_fields(payload)
    payload_json = json.dumps(cleaned_payload)
    headers = {"Content-Type": "application/json"}

    response = await retry_request(
        client.post, url, content=payload_json, headers=headers
    )
    response_json = response.json()
    logging.debug(f"Response JSON: {json.dumps(response_json, indent=2)}")

    return response_json


async def check_async_job_status(
    client: httpx.AsyncClient, job_url: str
) -> Optional[dict]:
    """Check the status of an asynchronous job.

    :param client: HTTPX async client instance.
    :param job_url: URL to check the job status.
    :return: JSON dictionary of the final response if completed, otherwise None.
    """
    while True:
        response = await retry_request(client.get, job_url)
        logging.debug(f"Job status response: {response.status_code} - {response.text}")
        status_response = StatusResponse.parse_obj(response.json())
        status = status_response.status
        description = (
            status_response.logs[0].message
            if status_response.logs
            else "No description provided"
        )

        if status == "Completed" and status_response.response_url:
            return await fetch_final_results(client, status_response.response_url)
        elif status in ["Failed", "Error"]:
            logging.error(f"Job failed with response: {status_response}")
            raise Exception(f"Job failed: {description}")
        elif status in ["Pending", "Running", "Queued"]:
            logging.info(f"Job is still in progress: {description}")
            await asyncio.sleep(5)
        else:
            logging.error(f"Unknown status: {status}. Description: {description}")
            return None


async def fetch_final_results(client: httpx.AsyncClient, response_url: str) -> dict:
    """Fetch the final results of an asynchronous job.

    :param client: HTTPX async client instance.
    :param response_url: URL to fetch the final results.
    :return: JSON dictionary response.
    """
    response = await retry_request(client.get, response_url)

    try:
        response_json = response.json()
    except ValueError as e:
        logging.error(
            f"Failed to parse final response JSON for URL {response_url}: {e}"
        )
        return None
    return response_json


async def process_query(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    curie: str,
    query_dict: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Process an individual query.

    :param client: HTTPX async client instance.
    :param semaphore: Semaphore to limit concurrent requests.
    :param curie: CURIE identifier for the query.
    :param query_dict: Dictionary representation of the query.
    :param index: Index of the query.
    :return: Dictionary of final results or None.
    """
    async with semaphore:
        try:
            query = Query.parse_obj(query_dict)
            initial_response_data = await post_async_query(
                client, ASYNC_QUERY_URL, query.dict()
            )

            if not isinstance(initial_response_data, dict):
                logging.error(
                    f"Initial response data is not a dictionary for curie {curie}: {initial_response_data}"
                )
                return None

            initial_response = InitialResponse(**initial_response_data)

            if not initial_response.job_url:
                logging.error(
                    f"No job_url found in the initial response for curie {curie}: {initial_response.dict()}"
                )
                return None

            final_response_data = await check_async_job_status(
                client, initial_response.job_url
            )

            if final_response_data:
                try:
                    final_response = FinalResponse(**final_response_data)
                except ValidationError as e:
                    print(final_response_data)
                    logging.error(
                        f"Validation error in final response for curie {curie}, detail: {e.errors()}"
                    )
                    return None

                final_result = final_response.dict()
                final_result["curie"] = curie
                final_result["index"] = index
                return final_result
            else:
                logging.error(f"Final response fetching failed for curie {curie}")
                return None
        except Exception as e:
            logging.error(
                f"Error while processing query {index} for curie {curie}: {e}"
            )
            return None


async def fetch_results_async(queries: List[Dict]) -> List[Dict]:
    """Fetch results for a list of queries asynchronously.

    :param queries: List of query dictionaries.
    :return: List of result dictionaries.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS),
        timeout=timeout_config,
    ) as client:
        tasks = [
            process_query(
                client, semaphore, query["curie"], query["query"], query["index"]
            )
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

        return [result for result in results if result]


def fetch_results(queries: List[Dict]) -> List[Dict]:
    """Synchronous wrapper for fetch_results_async.

    :param queries: List of query dictionaries.
    :return: List of result dictionaries.
    """
    return asyncio.run(fetch_results_async(queries))
