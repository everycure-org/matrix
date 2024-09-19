"""This module handles asynchronous query processing for the BTE pipeline."""

import asyncio
import logging
import httpx
import json
import time
import pandas as pd
from typing import List, Dict, Optional, Any, Callable

MAX_CONCURRENT_REQUESTS = 8  # maximum number of concurrent requests allowed
TOTAL_TIMEOUT = 310  # total timeout configuration for the entire request
CONNECT_TIMEOUT = 30.0  # timeout configuration for establishing a connection
READ_TIMEOUT = 250.0  # timeout configuration for HTTP requests
ASYNC_QUERY_URL = (
    "http://localhost:3000/v1/asyncquery"  # URL for asynchronous query processing
)
# ASYNC_QUERY_URL = "https://bte.hueb.org/v1/asyncquery"
MAX_RETRIES = 8  # maximum number of times to retry a request
RETRY_DELAY = 4  # time to wait before retrying a request
RETRY_BACKOFF_FACTOR = 2  # exponential backoff factor for retries
MAX_QUERY_RETRIES = 5  # maximum number of times to retry a query
QUERY_RETRY_DELAY = 4  # time to wait before retrying a failed query
JOB_CHECK_SLEEP = 0  # time to sleep between job status checks
DEBUG_QUERY_LIMITER = 8  # change to -1 to disable the limit
DEBUG_CSV_PATH = (
    "../../../predictions.csv"  # path to the output CSV file for debugging purposes
)

timeout_config = httpx.Timeout(
    timeout=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT
)


def generate_trapi_query(curie: str) -> Dict[str, Any]:
    """Generate a TRAPI query for a given CURIE.

    :param curie: The CURIE for which to generate the query.
    :return: A dictionary representing the TRAPI query object.
    :raises ValueError: If the CURIE is invalid.
    """
    if not curie:
        logging.error(f"Invalid curie: {curie}")
        raise ValueError("CURIE must not be empty")

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


async def generate_queries() -> List[Dict[str, Any]]:
    """Generate list of TRAPI queries from a DataFrame of diseases.

    :return: List of dictionaries, each containing 'curie', 'query', and 'index' keys corresponding to the CURIE, its TRAPI query, and its position in the DataFrame.
    """
    diseases = pd.read_csv(
        "https://github.com/everycure-org/matrix-disease-list/releases/latest/download/matrix-disease-list.tsv",
        sep="\t",
    )
    if DEBUG_QUERY_LIMITER > 0:
        diseases = diseases.iloc[:DEBUG_QUERY_LIMITER]

    queries = []
    for index, row in diseases.iterrows():
        curie = row.get("category_class")
        if curie:
            trapi_query = generate_trapi_query(curie)
            queries.append({"curie": curie, "query": trapi_query, "index": index})

    logging.info(f"Generated {len(queries)} queries")
    return queries


async def transform_result(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform a single raw query result into a list of dictionaries with target, source, and score.

    :param response: The raw response from the query.
    :return: List of dictionaries with keys 'target', 'source', and 'score'.
    """
    results = []
    if not response:
        logging.warning("Received empty response.")
        return results

    curie = response.get("curie")
    logging.info(f"Transforming results for curie {curie}")

    if "message" not in response or not response["message"]:
        logging.error(f"Message is missing in the response for curie {curie}")
        return results

    if not response["message"].get("results"):
        logging.warning(f"No results found in the message for curie {curie}")
        return results

    for result in response["message"]["results"]:
        node_bindings = result.get("node_bindings", {})
        n1_node_bindings = node_bindings.get("n1", [])
        n0_bindings = node_bindings.get("n0", [])
        if not n1_node_bindings:
            logging.error(f"n1 node_bindings missing in result for curie {curie}")
            continue

        n0_bindings = result.get("node_bindings", {}).get("n0", [])
        analyses = result.get("analyses", [])

        if not (len(n1_node_bindings) == len(n0_bindings) == len(analyses)):
            logging.warning(
                f"Mismatched lengths of bindings and analyses for curie {curie}"
            )
            continue

        for n1_binding, n0_binding, analysis in zip(
            n1_node_bindings, n0_bindings, analyses
        ):
            result_dict = {
                "target": n1_binding.get("id"),
                "source": n0_binding.get("id"),
                "score": analysis.get("score"),
            }
            logging.debug(f"Transformed result: {result_dict}")
            results.append(result_dict)

    logging.info(f"Transformed {len(results)} results for curie {curie}")
    return results


async def retry_request(
    request_func: Callable,
    *args,
    retries: int = MAX_RETRIES,
    delay: int = RETRY_DELAY,
    **kwargs,
) -> httpx.Response:
    """Perform an asynchronous HTTP request with retry logic.

    :param request_func: Asynchronous function for making the HTTP request (e.g., client.get or client.post).
    :param args: Positional arguments to pass to the request function.
    :param retries: Number of times to retry the request in case of failure. Default is 8.
    :param delay: Initial delay in seconds between retries. The delay is multiplied by the backoff factor with each subsequent retry. Default is 4.
    :param kwargs: Keyword arguments to pass to the request function.
    :return: HTTPX Response object.
    :raises: Raises an HTTPStatusError or RequestError if all retry attempts fail.
    """
    for attempt in range(retries):
        try:
            response = await request_func(*args, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logging.error(f"Error: {e}. Attempt {attempt + 1} of {retries}")
            if attempt < retries - 1:
                logging.debug(
                    f"Retrying the request. Next attempt in {delay * (RETRY_BACKOFF_FACTOR ** attempt)} seconds."
                )
                await asyncio.sleep(delay * (RETRY_BACKOFF_FACTOR**attempt))
            else:
                logging.error("Max retries exceeded.")
                raise


async def post_async_query(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """Post an asynchronous query to the specified URL.

    :param client: HTTPX async client instance.
    :param url: URL to which the query is posted.
    :param payload: Payload of the query.
    :return: A dictionary parsed from the JSON response.
    """
    logging.info(
        f"Submitting query to {url} with payload: {json.dumps(payload, indent=2)}"
    )
    response = await retry_request(client.post, url, json=payload)
    response_json = response.json()
    return response_json


async def check_async_job_status(
    client: httpx.AsyncClient, job_url: str
) -> Optional[dict]:
    """Check the status of an asynchronous job.

    :param client: HTTPX async client instance.
    :param job_url: URL to check the job status.
    :return:
        - A dictionary of the final response if the job is completed successfully.
        - A dictionary with 'status' and 'description' if the job fails, errors, or times out.
    """
    start_time = time.monotonic()

    while True:
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > TOTAL_TIMEOUT:
            logging.error(f"Job {job_url} timed out after {TOTAL_TIMEOUT} seconds.")
            return {
                "status": "TimedOut",
                "description": f"Job did not complete within {TOTAL_TIMEOUT} seconds.",
            }

        try:
            response = await retry_request(client.get, job_url)
            response_json = response.json()
            status = response_json.get("status", "Unknown")
            description = response_json.get(
                "description", "No status description provided."
            )
            logging.info(f"Job status for {job_url}: {status} - {description}")

            if status == "Completed" and response_json.get("response_url"):
                return await fetch_final_results(client, response_json["response_url"])
            elif status in ["Failed", "Error"]:
                logging.error(f"Job failed with response: {response_json}")
                return {"status": status, "description": description}
            elif status in ["Pending", "Running", "Queued"]:
                logging.info(f"Job is still in progress: {description}")
                await asyncio.sleep(JOB_CHECK_SLEEP)
            else:
                logging.error(f"Unknown status: {status}. Description: {description}")
                return {"status": status, "description": description}
        except Exception as e:
            logging.error(f"Exception while checking job status for {job_url}: {e}")
            await asyncio.sleep(QUERY_RETRY_DELAY)


async def fetch_final_results(client: httpx.AsyncClient, response_url: str) -> dict:
    """Fetch and parse the final results of an asynchronous job.

    :param client: HTTPX async client instance.
    :param response_url: URL to retrieve the final response.
    :return: A dictionary parsed from the JSON final response.
    :raises Exception: If the response JSON cannot be parsed.
    """
    logging.info(f"Fetching final results from {response_url}")
    response = await retry_request(client.get, response_url)

    try:
        response_json = response.json()
        return response_json
    except ValueError as e:
        logging.error(
            f"Failed to parse final response JSON for URL {response_url}: {e}"
        )
        raise Exception("Failed to parse JSON")


async def process_query(
    client: httpx.AsyncClient,
    curie: str,
    query_dict: Dict[str, Any],
    index: int,
    result_queue: asyncio.Queue,
):
    """Process an individual query, sending results to the result_queue.

    :param client: HTTPX async client instance.
    :param curie: CURIE of the query.
    :param query_dict: Dictionary containing the query.
    :param index: Index of the query.
    :param result_queue: Queue to put the results of processed queries.
    :raises: This function handles its own exceptions and does not raise them to the caller.
    """
    for attempt in range(1, MAX_QUERY_RETRIES + 1):
        try:
            logging.info(f"Attempt {attempt} for query {index} (CURIE: {curie})")
            initial_response_data = await post_async_query(
                client, ASYNC_QUERY_URL, query_dict
            )
            if not isinstance(
                initial_response_data, dict
            ) or not initial_response_data.get("job_url"):
                logging.error(
                    f"Invalid initial response for CURIE {curie}: {initial_response_data}"
                )
                raise ValueError("Invalid initial response structure.")

            final_response_data = await check_async_job_status(
                client, initial_response_data["job_url"]
            )

            if final_response_data and final_response_data.get("status") not in [
                "Failed",
                "Error",
            ]:
                final_response_data.update({"curie": curie, "index": index})
                await result_queue.put(final_response_data)
                logging.info(f"Successfully processed query {index} for CURIE {curie}")
                return
            else:
                status = (
                    final_response_data.get("status")
                    if final_response_data
                    else "Unknown"
                )
                logging.error(
                    f"Query {index} for CURIE {curie} failed with status: {status}. Retrying..."
                )
                await asyncio.sleep(QUERY_RETRY_DELAY)
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as e:
            logging.error(
                f"Error on attempt {attempt} for query {index} (CURIE: {curie}): {e}"
            )
            if attempt < MAX_QUERY_RETRIES:
                logging.info(f"Retrying query {index} for CURIE {curie} after error.")
                await asyncio.sleep(QUERY_RETRY_DELAY)
            else:
                logging.error(
                    f"Maximum retries reached for query {index} (CURIE: {curie})."
                )
        except KeyError as e:
            logging.exception(
                f"Missing expected key {e} in response for query {index} (CURIE: {curie})."
            )
            if attempt < MAX_QUERY_RETRIES:
                logging.info(
                    f"Retrying query {index} for CURIE {curie} due to missing key."
                )
                await asyncio.sleep(QUERY_RETRY_DELAY)
            else:
                logging.error(
                    f"Maximum retries reached for query {index} (CURIE: {curie}) due to missing key."
                )
        except Exception as e:
            logging.exception(
                f"Unexpected error on attempt {attempt} for query {index} (CURIE: {curie}): {e}"
            )
            if attempt < MAX_QUERY_RETRIES:
                logging.info(
                    f"Retrying query {index} for CURIE {curie} after unexpected error."
                )
                await asyncio.sleep(QUERY_RETRY_DELAY)
            else:
                logging.error(
                    f"Maximum retries reached for query {index} (CURIE: {curie}) due to unexpected error."
                )

    logging.error(
        f"Failed to process query {index} for CURIE {curie} after {MAX_QUERY_RETRIES} attempts."
    )


async def producer(query_queue: asyncio.Queue, num_consumers: int):
    """Producer task to generate and enqueue queries.

    :param query_queue: Queue to enqueue the generated queries.
    :param num_consumers: Number of consumer tasks to signal completion.
    """
    queries = await generate_queries()
    for query in queries:
        await query_queue.put(query)

    for _ in range(num_consumers):
        await query_queue.put(None)
    logging.info("Producer has put all queries and exit signals into the queue.")


async def consumer(
    query_queue: asyncio.Queue, result_queue: asyncio.Queue, client: httpx.AsyncClient
):
    """Consumer task to process queries from the query queue and output results.

    :param query_queue: Queue from which to get queries to process.
    :param result_queue: Queue to put the results of processed queries.
    :param client: HTTPX async client instance.
    """
    while True:
        query = await query_queue.get()
        if query is None:
            query_queue.task_done()
            break

        try:
            logging.info(
                f"Consumer started processing query {query['index']} for curie {query['curie']}"
            )
            await process_query(
                client, query["curie"], query["query"], query["index"], result_queue
            )
            logging.info(
                f"Consumer finished processing query {query['index']} for curie {query['curie']}"
            )
        except Exception as e:
            logging.exception(
                f"Error processing query {query.get('index', 'Unknown')}: {e}"
            )
        finally:
            query_queue.task_done()
            if query is not None:
                logging.info(
                    f"query_queue task_done called for query {query['index']}."
                )

    await result_queue.put(None)
    logging.info("Exiting consumer loop.")


async def stream_results(semaphore, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Continuously stream results for TRAPI queries.

    :param semaphore: An asyncio.Semaphore instance to limit the number of concurrent requests.
    :param client: HTTPX async client instance.
    :return: List of transformed query results.
    """
    async with semaphore:
        results = []
        query_queue = asyncio.Queue()
        result_queue = asyncio.Queue()

        num_consumers = MAX_CONCURRENT_REQUESTS

        producer_task = asyncio.create_task(producer(query_queue, num_consumers))
        logging.info("Producer task created.")

        consumer_tasks = [
            asyncio.create_task(consumer(query_queue, result_queue, client))
            for _ in range(num_consumers)
        ]
        logging.info("Consumer tasks created.")

        completion_signals_received = 0

        while completion_signals_received < num_consumers:
            result = await result_queue.get()
            if result is None:
                completion_signals_received += 1
                logging.info(
                    f"Received completion signal from consumer. Total received: {completion_signals_received}"
                )
                result_queue.task_done()
                continue

            logging.info(
                f"Received result for curie {result['curie']}, starting transformation."
            )
            transformed_results = await transform_result(result)
            if transformed_results:
                results.extend(transformed_results)
                logging.info(
                    f"Results transformed and added for curie {result['curie']}"
                )
            result_queue.task_done()

        logging.info(
            f"All consumers have signaled completion. Collected {len(results)} results in total."
        )

        await producer_task
        await query_queue.join()
        for consumer_task in consumer_tasks:
            await consumer_task

        return results


async def async_bte_kedro_node_function() -> pd.DataFrame:
    """Node function to handle async processing within a Kedro pipeline node.

    :return: DataFrame containing the final query results with columns 'target', 'source', and 'score'.
    """
    logging.info("Starting async processing for BTE Kedro node function")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        results = await stream_results(semaphore, client)

    logging.info(f"Total results collected: {len(results)}")
    if results:
        logging.debug(f"Sample results: {results[:5]}")
    else:
        logging.warning("No results were collected during asynchronous processing.")

    df = pd.DataFrame(results, columns=["target", "source", "score"])

    logging.info("Async processing completed for BTE Kedro node function")
    logging.info(f"DataFrame head:\n{df.head()}")
    logging.info(f"Total rows in DataFrame: {len(df)}")

    return df


def bte_kedro_node_function() -> pd.DataFrame:
    """Wrapper to run the stream processing function synchronously using asyncio.run.

    :return: DataFrame containing the final query results with columns 'target', 'source', and 'score'.
    :side-effect: Saves the DataFrame to a CSV file at the specified path for debugging.
    """
    df = asyncio.run(async_bte_kedro_node_function())

    # save_dataframe_to_csv(df)
    return df


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str = DEBUG_CSV_PATH) -> None:
    """Save the provided DataFrame to a CSV file at the specified path.

    :param df: DataFrame to be saved.
    :param file_path: Path where the DataFrame will be saved as a CSV file.
    :raises ValueError: If the provided DataFrame is empty.
    :raises FileNotFoundError: If the specified file path does not exist.
    :raises PermissionError: If there are insufficient permissions to write to the specified path.
    """
    if df.empty:
        logging.error("The provided DataFrame is empty.")
        raise ValueError("The DataFrame to save is empty.")

    try:
        df.to_csv(file_path, index=False)
        logging.info(f"DataFrame successfully saved to {file_path}")
    except (FileNotFoundError, PermissionError) as e:
        logging.exception(f"Failed to save DataFrame to {file_path}: {e}")
        raise
