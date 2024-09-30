"""This module handles asynchronous query processing for the BTE pipeline."""
# NOTE: This file was partially generated using AI assistance.

import asyncio
import logging
import httpx
import time
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from typing import Any, Iterator, Optional
from refit.v1.core.inline_has_schema import has_schema


def generate_trapi_query(curie: str) -> dict[str, Any]:
    """Generate a TRAPI query for a given CURIE.

    Args:
        curie (str): The CURIE identifier for which to generate the query.

    Returns:
        dict[str, Any]: A dictionary representing the TRAPI query.
    """
    if not curie:
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
    disease_list: pd.DataFrame, n_diseases_limit: int
) -> Iterator[dict]:
    """Generate TRAPI queries from the disease list.

    Args:
        disease_list (pd.DataFrame): A DataFrame containing the list of diseases.
        n_diseases_limit (int): The maximum number of diseases to process.

    Yields:
        Iterator[dict[str, Any]]: An iterator of dictionaries containing the CURIE and the TRAPI query.
    """
    if n_diseases_limit > 0:
        disease_list = disease_list.head(n_diseases_limit)
    for index, row in disease_list.iterrows():
        curie = row.get("category_class")
        if curie:
            trapi_query = generate_trapi_query(curie)
            yield {"curie": curie, "query": trapi_query, "index": index}


def transform_result(
    response: dict[str, Any],
    drug_set1: set,
    drug_set2: set,
    drug_mapping_dict: dict[str, str],
) -> list[dict[str, Any]]:
    """Transform the raw query response into a list of result dictionaries.

    Args:
        response (dict[str, Any]): The raw response from the query.
        drug_set1 (set): Set containing drug CURIEs to be used for determining approval status.
        drug_set2 (set): Set containing drug single_IDs to be used for determining approval status.
        drug_mapping_dict (dict[str, str]): A dictionary mapping drug CURIEs to their corresponding synonymized IDs.

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing transformed results.
    """
    results = []
    curie = response.get("curie")
    message = response.get("message", {})

    if not message.get("results"):
        return results

    # Iterate over the results and extract source, target, score
    for result in message["results"]:
        node_bindings = result.get("node_bindings", {})
        n1_bindings = node_bindings.get("n1", [])
        n0_bindings = node_bindings.get("n0", [])
        analyses = result.get("analyses", [])

        for n1_binding, n0_binding, analysis in zip(n1_bindings, n0_bindings, analyses):
            drug_curie = n0_binding.get("id")

            if drug_curie in drug_set1:
                approved = True
            elif drug_curie in drug_set2:
                approved = True
                drug_curie = drug_mapping_dict[drug_curie]
            else:
                approved = False

            results.append(
                {
                    "target": n1_binding.get("id"),
                    "source": drug_curie,
                    "score": analysis.get("score"),
                    "approved": approved,
                }
            )

    return results


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=4, max=60),
    reraise=True,
)
async def post_async_query(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """Post an asynchronous query to the specified URL.

    Args:
        client (httpx.AsyncClient): The HTTP client for making requests.
        url (str): The URL to which the query should be posted.
        payload (dict): The payload of the query.

    Returns:
        dict: The JSON response from the server.
    """
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

    Args:
        client (httpx.AsyncClient): The HTTP client for making requests.
        response_url (str): The URL from which to fetch the final results.

    Returns:
        dict: The parsed JSON response containing the final results.
    """
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

    Args:
        client (httpx.AsyncClient): The HTTP client for making requests.
        job_url (str): The URL to check the job status.
        default_timeout (int): The maximum time to wait for the job to complete.
        job_check_sleep (float): The time to wait between status checks.

    Returns:
        Optional[dict]: The final results if the job is completed, otherwise None.
    """
    start_time = time.monotonic()
    while True:
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > default_timeout:
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

            if status == "Completed" and response_json.get("response_url"):
                return await fetch_final_results(client, response_json["response_url"])
            elif status in ["Failed", "Error"]:
                return {"status": status, "description": description}
            elif status in ["Pending", "Running", "Queued"]:
                await asyncio.sleep(job_check_sleep)
            else:
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
    drug_set1: set,
    drug_set2: set,
    drug_mapping_dict: dict[str, str],
) -> Optional[list[dict[str, Any]]]:
    """Process an individual query and return transformed results.

    Args:
        client (httpx.AsyncClient): The HTTP client for making requests.
        query (dict[str, Any]): The query to process.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.
        async_query_url (str): The URL to submit the asynchronous query.
        default_timeout (int): The maximum time to wait for the job to complete.
        job_check_sleep (float): The time to wait between status checks.
        drug_set1 (set): Set containing drug CURIEs to be used for determining approval status.
        drug_set2 (set): Set containing drug single_IDs to be used for determining approval status.
        drug_mapping_dict (dict[str, str]): A dictionary mapping drug single_IDs to their corresponding synonymized CURIE IDs.

    Returns:
        Optional[list[dict[str, Any]]]: A list of transformed results if successful, otherwise None.
    """
    curie = query["curie"]
    query_dict = query["query"]
    index = query["index"]

    async with semaphore:
        initial_response_data = await post_async_query(
            client, async_query_url, query_dict
        )
        if not initial_response_data.get("job_url"):
            raise ValueError("Invalid initial response structure.")

        final_response_data = await check_async_job_status(
            client, initial_response_data["job_url"], default_timeout, job_check_sleep
        )

        if final_response_data and final_response_data.get("status") in [
            "Completed",
            "Success",
        ]:
            final_response_data.update({"curie": curie, "index": index})
            transformed_results = transform_result(
                final_response_data, drug_set1, drug_set2, drug_mapping_dict
            )
            return transformed_results
        else:
            status = (
                final_response_data.get("status", "Unknown")
                if final_response_data
                else "Unknown"
            )
            raise Exception(f"Query failed with status: {status}")


async def run_async_queries(
    disease_list: pd.DataFrame,
    drug_set1: set,
    drug_set2: set,
    drug_mapping_dict: dict,
    max_concurrent_requests: int,
    async_query_url: str,
    default_timeout: int,
    job_check_sleep: float,
    n_diseases_limit: int,
) -> pd.DataFrame:
    """Run asynchronous query processing and return a DataFrame of results.

    Args:
        disease_list (pd.DataFrame): DataFrame containing the list of diseases to query.
        drug_set1 (set): Set containing drug CURIEs to be used for determining approval status.
        drug_set2 (set): Set containing drug single_IDs to be used for determining approval status.
        drug_mapping_dict (dict[str, str]): A dictionary mapping drug single_IDs to their corresponding synonymized CURIE IDs.
        max_concurrent_requests (int): The maximum number of concurrent queries to allow.
        async_query_url (str): The URL to submit the asynchronous query.
        default_timeout (int): The maximum time to wait for the job to complete.
        job_check_sleep (float): The time to wait between status checks.
        n_diseases_limit (int): The maximum number of diseases to process.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the queries.
    """
    timeout_config = httpx.Timeout(default_timeout)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        queries = list(generate_queries(disease_list, n_diseases_limit))
        tasks = [
            asyncio.create_task(
                process_query(
                    client,
                    query,
                    semaphore,
                    async_query_url,
                    default_timeout,
                    job_check_sleep,
                    drug_set1,
                    drug_set2,
                    drug_mapping_dict,
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

    schema = {
        "target": "object",
        "source": "object",
        "score": "float64",
        "approved": "bool",
    }
    df = pd.DataFrame({col: pd.Series(dtype=schema[col]) for col in schema})
    return df


@has_schema(
    schema={
        "target": "object",
        "source": "object",
        "score": "float64",
        "approved": "bool",
    },
    allow_subset=True,
)
def run_bte_queries(
    disease_list: pd.DataFrame,
    drug_list: pd.DataFrame,
    async_query_url: str,
    max_concurrent_requests: int,
    default_timeout: int,
    job_check_sleep: float,
    n_diseases_limit: int,
) -> pd.DataFrame:
    """Synchronous wrapper for running asynchronous queries.

    Args:
        disease_list (pd.DataFrame): DataFrame containing the list of diseases to query.
        drug_list (pd.DataFrame): DataFrame containing the list of approved drugs.
        max_concurrent_requests (int): The maximum number of concurrent queries to allow.
        async_query_url (str): The URL to submit the asynchronous query.
        default_timeout (int): The maximum time to wait for the job to complete.
        job_check_sleep (float): The time to wait between status checks.
        n_diseases_limit (int): The maximum number of diseases to process.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the queries.
    """
    drug_set1 = set(drug_list["curie"])
    drug_set2 = set(drug_list["single_ID"])
    drug_mapping_dict = dict(zip(drug_list["single_ID"], drug_list["curie"]))

    df = asyncio.run(
        run_async_queries(
            disease_list,
            drug_set1,
            drug_set2,
            drug_mapping_dict,
            max_concurrent_requests,
            async_query_url,
            default_timeout,
            job_check_sleep,
            n_diseases_limit,
        )
    )
    return df
