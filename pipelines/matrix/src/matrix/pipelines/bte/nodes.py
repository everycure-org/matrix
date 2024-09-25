"""This module handles asynchronous query processing for the BTE pipeline."""
# NOTE: This file was partially generated using AI assistance.

import asyncio
import logging
import httpx
import time
import itertools
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from typing import Any, Iterator, Optional

# # Configuration constants for managing asynchronous requests and retries
# MAX_CONCURRENT_REQUESTS = 12  # Maximum number of concurrent requests allowed
# default_timeout = 310  # Total timeout configuration for the entire request
# CONNECT_TIMEOUT = 30.0  # Timeout configuration for establishing a connection
# READ_TIMEOUT = 310.0  # Timeout configuration for HTTP requests
# ASYNC_QUERY_URL = (
#     "http://localhost:3000/v1/asyncquery"  # URL for asynchronous query processing
# )
# MAX_RETRIES = 8  # Maximum number of times to retry a request
# RETRY_DELAY = 4  # Time to wait before retrying a request
# RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor for retries
# MAX_QUERY_RETRIES = 5  # Maximum number of times to retry a query
# QUERY_RETRY_DELAY = 4  # Time to wait before retrying a failed query
# JOB_CHECK_SLEEP = 0.5  # Time to sleep between job status checks
# DEBUG_QUERY_LIMITER = 8  # Change to -1 to disable the limit
# # DISEASE_LIST = pd.read_csv(
# #     "https://github.com/everycure-org/matrix-disease-list/releases/latest/download/matrix-disease-list.tsv",
# #     sep="\t",
# # )  # Load the disease list from a remote TSV file
# DISEASE_LIST = pd.DataFrame
# DEBUG_CSV_PATH = (
#     "../../../predictions.csv"  # Path to the output CSV file for debugging purposes
# )

# # Configure timeout settings for HTTP requests
# timeout_config = httpx.Timeout(
#     timeout=default_timeout, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT
# )


def generate_trapi_query(curie: str) -> dict[str, Any]:
    """Generate a TRAPI query for a given CURIE.

    Parameters:
    curie (str): The CURIE identifier for which to generate the query.

    Returns:
    dict[str, Any]: A dictionary representing the TRAPI query.
    """
    if not curie:
        logging.error("CURIE must not be empty")
        raise ValueError("CURIE must not be empty")

    # Construct the TRAPI query structure
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:ChemicalEntity"]},
                    "n1": {
                        "ids": [curie],
                        "categories": ["biolink:DiseaseOrPhenotypicFeature"],
                    },
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:treats"],
                        "knowledge_type": "inferred",
                    }
                },
            }
        }
    }


def generate_queries(
    disease_list: pd.DataFrame, debug_query_limiter: int
) -> Iterator[dict]:
    """Generate TRAPI queries from the disease list.

    Yields:
    Iterator[dict[str, Any]]: An iterator of dictionaries containing the CURIE and the TRAPI query.
    """
    for index, row in itertools.islice(disease_list.iterrows(), debug_query_limiter):
        curie = row.get("category_class")
        if curie:
            trapi_query = generate_trapi_query(curie)
            yield {"curie": curie, "query": trapi_query, "index": index}


def transform_result(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform the raw query response into a list of result dictionaries.

    Parameters:
    response (dict[str, Any]): The raw response from the query.

    Returns:
    list[dict[str, Any]]: A list of dictionaries containing transformed results.
    """
    results = []
    curie = response.get("curie")

    message = response.get("message", {})
    if not message.get("results"):
        logging.debug(f"No results found for CURIE {curie}")
        return results

    # Iterate over the results and extract source, target, score
    for result in message["results"]:
        node_bindings = result.get("node_bindings", {})
        n1_bindings = node_bindings.get("n1", [])
        n0_bindings = node_bindings.get("n0", [])
        analyses = result.get("analyses", [])

        if not (n1_bindings and n0_bindings and analyses):
            logging.warning(f"Incomplete data in result for CURIE {curie}")
            continue

        for n1_binding, n0_binding, analysis in zip(n1_bindings, n0_bindings, analyses):
            results.append(
                {
                    "target": n1_binding.get("id"),
                    "source": n0_binding.get("id"),
                    "score": analysis.get("score"),
                }
            )

    logging.info(f"Transformed {len(results)} results for CURIE {curie}")
    return results


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=4, max=60),
    reraise=True,
)
async def post_async_query(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """Post an asynchronous query to the specified URL.

    Parameters:
    client (httpx.AsyncClient): The HTTP client for making requests.
    url (str): The URL to which the query should be posted.
    payload (dict): The payload of the query.

    Returns:
    dict: The JSON response from the server.
    """
    logging.debug(f"Submitting query to {url} with payload: {payload}")
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


@retry(
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.RequestError, ValueError)
    ),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=4, max=60),
    reraise=True,
)
async def fetch_final_results(client: httpx.AsyncClient, response_url: str) -> dict:
    """Fetch and parse the final results of an asynchronous job.

    Parameters:
    client (httpx.AsyncClient): The HTTP client for making requests.
    response_url (str): The URL from which to fetch the final results.

    Returns:
    dict: The parsed JSON response containing the final results.
    """
    logging.debug(f"Fetching final results from {response_url}")
    response = await client.get(response_url)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as e:
        logging.error(f"Failed to parse JSON from {response_url}: {e}")
        raise


async def check_async_job_status(
    client: httpx.AsyncClient,
    job_url: str,
    default_timeout: int,
    job_check_sleep: float,
) -> Optional[dict]:
    """Check the status of an asynchronous job and fetch final results if completed.

    Parameters:
    client (httpx.AsyncClient): The HTTP client for making requests.
    job_url (str): The URL to check the job status.

    Returns:
    Optional[dict]: The final results if the job is completed, otherwise None.
    """
    start_time = time.monotonic()
    while True:
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > default_timeout:
            logging.error(f"Job {job_url} timed out after {default_timeout} seconds.")
            return {
                "status": "TimedOut",
                "description": f"Job did not complete within {default_timeout} seconds.",
            }

        try:
            response = await client.get(job_url)
            response.raise_for_status()
            response_json = response.json()
            status = response_json.get("status", "Unknown")
            description = response_json.get(
                "description", "No status description provided."
            )
            logging.debug(f"Job status for {job_url}: {status} - {description}")

            if status == "Completed" and response_json.get("response_url"):
                return await fetch_final_results(client, response_json["response_url"])
            elif status in ["Failed", "Error"]:
                logging.error(f"Job failed with response: {response_json}")
                return {"status": status, "description": description}
            elif status in ["Pending", "Running", "Queued"]:
                logging.debug(f"Job is still in progress: {description}")
                await asyncio.sleep(job_check_sleep)
            else:
                logging.error(f"Unknown status: {status}. Description: {description}")
                return {"status": status, "description": description}
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
            logging.exception(f"Error while checking job status for {job_url}: {e}")
            await asyncio.sleep(job_check_sleep)
        except Exception as e:
            logging.exception(
                f"Unexpected error while checking job status for {job_url}: {e}"
            )
            await asyncio.sleep(job_check_sleep)


@retry(
    retry=retry_if_exception_type(
        (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError, Exception)
    ),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=4, max=60),
    reraise=True,
)
async def process_query(
    client: httpx.AsyncClient,
    query: dict[str, Any],
    semaphore: asyncio.Semaphore,
    async_query_url: str,
    default_timeout: int,
    job_check_sleep: float,
) -> Optional[list[dict[str, Any]]]:
    """Process an individual query and return transformed results.

    Parameters:
    client (httpx.AsyncClient): The HTTP client for making requests.
    query (dict[str, Any]): The query to process.
    semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
    Optional[list[dict[str, Any]]]: A list of transformed results if successful, otherwise None.
    """
    curie = query["curie"]
    query_dict = query["query"]
    index = query["index"]

    async with semaphore:
        logging.debug(f"Processing query {index} for CURIE {curie}")
        initial_response_data = await post_async_query(
            client, async_query_url, query_dict
        )
        if not initial_response_data.get("job_url"):
            logging.error(f"Invalid initial response for CURIE {curie}")
            raise ValueError("Invalid initial response structure.")

        final_response_data = await check_async_job_status(
            client, initial_response_data["job_url"], default_timeout, job_check_sleep
        )

        if final_response_data and final_response_data.get("status") in [
            "Completed",
            "Success",
        ]:
            final_response_data.update({"curie": curie, "index": index})
            transformed_results = transform_result(final_response_data)
            logging.debug(f"Successfully processed query {index} for CURIE {curie}")
            return transformed_results
        else:
            status = (
                final_response_data.get("status", "Unknown")
                if final_response_data
                else "Unknown"
            )
            logging.error(
                f"Query {index} for CURIE {curie} failed with status: {status}."
            )
            raise Exception(f"Query failed with status: {status}")


async def run_async_queries(
    disease_list: pd.DataFrame,
    max_concurrent_requests: int,
    async_query_url: str,
    default_timeout: int,
    job_check_sleep: float,
    debug_query_limiter: int,
) -> pd.DataFrame:
    """Run asynchronous query processing and return a DataFrame of results.

    Returns:
    pd.DataFrame: A DataFrame containing the results of the queries.
    """
    timeout_config = httpx.Timeout(default_timeout)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        queries = list(generate_queries(disease_list, debug_query_limiter))
        tasks = [
            asyncio.create_task(
                process_query(
                    client,
                    query,
                    semaphore,
                    async_query_url,
                    default_timeout,
                    job_check_sleep,
                )
            )
            for query in queries
        ]
        all_results = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                if result:
                    all_results.extend(result)
            except Exception as e:
                logging.error(f"An error occurred during query processing: {e}")

    logging.debug(f"Total results collected: {len(all_results)}")

    df = pd.DataFrame(all_results, columns=["target", "source", "score"])
    return df


def run_bte_queries(
    disease_list: pd.DataFrame,
    async_query_url: str,
    max_concurrent_requests: int,
    default_timeout: int,
    job_check_sleep: float,
    debug_query_limiter: int,
    debug_csv_path: str,
) -> pd.DataFrame:
    """Synchronous wrapper for running asynchronous queries.

    Parameters:
    disease_list (pd.DataFrame): The DataFrame containing the disease list to use.

    Returns:
    pd.DataFrame: A DataFrame containing the results of the queries.
    """
    print(disease_list.head)
    df = asyncio.run(
        run_async_queries(
            disease_list,
            max_concurrent_requests,
            async_query_url,
            default_timeout,
            job_check_sleep,
            debug_query_limiter,
        )
    )
    save_dataframe_to_csv(df, debug_csv_path)
    return df


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """Save the provided DataFrame to a CSV file at the specified path.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The file path where the DataFrame should be saved.

    Raises:
    ValueError: If the DataFrame is empty.
    """
    if df.empty:
        logging.error("The provided DataFrame is empty.")
        raise ValueError("The DataFrame to save is empty.")

    try:
        df.to_csv(file_path, index=False)
        logging.debug(f"DataFrame successfully saved to {file_path}")
    except (FileNotFoundError, PermissionError) as e:
        logging.exception(f"Failed to save DataFrame to {file_path}: {e}")
        raise
