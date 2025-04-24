import logging
from typing import Dict, List

from tenacity import stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

import asyncio
from typing import Dict, List

import aiohttp
import nest_asyncio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

nest_asyncio.apply()


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def process_batch(batch: List[str], session: aiohttp.ClientSession, pbar: tqdm, url: str) -> Dict:
    """Process a single batch of IDs with retry logic"""
    results = {}

    # Filter out any empty or None values
    batch = [x for x in batch if x and isinstance(x, str)]

    payload = {"curies": batch, "conflate": True, "description": False, "drug_chemical_conflate": False}

    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                for curie, node_data in result.items():
                    if node_data:
                        results[curie] = {
                            "id": node_data["id"]["identifier"] if node_data.get("id") else None,
                            "label": node_data["id"].get("label", None) if node_data.get("id") else None,
                            "all_categories": node_data.get("type", None),
                            "original_id": curie,
                        }
                    else:
                        results[curie] = {"id": None, "label": None, "all_categories": None, "original_id": curie}
            elif response.status == 502:
                print(f"Server overloaded (502). Retrying batch after delay...")
                await asyncio.sleep(10)  # Longer delay for server errors
                raise aiohttp.ClientError("Server overloaded")
            elif response.status == 422:
                print(f"Invalid data in batch (422). Skipping problematic IDs...")
                print(f"Problematic batch: {batch}")
                return {}
            else:
                print(f"Error status {response.status}")

    except Exception as e:
        print(f"Error processing batch: {e}")
        raise

    pbar.update(1)
    return results


async def resolve_ids_batch_async(
    curies: List[str],
    batch_size: int = 250,
    max_concurrent: int = 10,
    url: str = "https://nodenorm.test.transltr.io/1.5/get_normalized_nodes",
) -> Dict:
    """
    Resolve IDs using concurrent async POST requests with improved error handling
    """
    all_results = {}

    # Clean input data
    curies = [str(x) for x in curies if x is not None]

    # Split into batches
    batches = [curies[i : i + batch_size] for i in range(0, len(curies), batch_size)]
    total_batches = len(batches)

    pbar = tqdm(total=total_batches, desc="Processing batches")

    connector = aiohttp.TCPConnector(
        limit=max_concurrent, force_close=False, enable_cleanup_closed=True, ttl_dns_cache=300
    )

    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(batch):
            async with semaphore:
                try:
                    return await process_batch(batch, session, pbar, url)
                except Exception as e:
                    print(f"Batch failed: {e}")
                    return {}

        for i in range(0, len(batches), max_concurrent):
            batch_group = batches[i : i + max_concurrent]
            tasks = [process_with_semaphore(batch) for batch in batch_group]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict):
                    all_results.update(result)

            await asyncio.sleep(1)

    pbar.close()
    return all_results
