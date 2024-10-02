"""Module for normalizing knowledge graph nodes using an renci API."""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import Dict, Any, List
from tqdm import tqdm
from joblib import Memory
from functools import reduce
import pyspark.sql.functions as f

memory = Memory(location=".cache/nodenorm", verbose=0)
nn_endpoint = f'{os.getenv("NODE_NORMALIZER_SERVER", "https://nodenormalization-sri.renci.org/")}/get_normalized_nodes'


def union_multiple_kg_nodes(**kwargs):
    """Combine multiple knowledge graph node dataframes into a single unified dataframe.

    Args:
        **kwargs: Keyword arguments where keys are KG names and values are PySpark DataFrames.

    Returns:
        PySpark DataFrame: A unified dataframe containing nodes from all input KGs.
    """
    kgs = list(kwargs.values())
    assert len(kgs) > 0, "No knowledge graphs provided"

    # Function to merge two dataframes
    def merge_kgs(df1, df2):
        return df1.unionByName(df2, allowMissingColumns=False)

    # Merge all KGs using reduce
    unified_kg = reduce(merge_kgs, kgs)

    # Combine array columns
    array_columns = [
        col for col, dtype in unified_kg.dtypes if dtype.startswith("array")
    ]
    for col in array_columns:
        unified_kg = unified_kg.groupBy("id").agg(f.collect_set(col).alias(col))

    return unified_kg


def union_multiple_kg_edges(**kwargs):
    """Combine multiple knowledge graph edge dataframes into a single unified dataframe.

    Args:
        **kwargs: Keyword arguments where keys are KG names and values are PySpark DataFrames.

    Returns:
        PySpark DataFrame: A unified dataframe containing edges from all input KGs.
    """
    # NOTE: edges are potentially much harder to merge than nodes
    # because edges can have different meanings across KG and contain
    # complex properties such as confidence, knowledge level, qualifiers etc.
    # This requires a conversation and then a good first shot at the problem.
    pass


def normalize_kg(nodes: ps.sql.DataFrame, edges: ps.sql.DataFrame):
    """Function normalizes a KG using external API endpoint.

    This functions takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with

    """
    node_ids = nodes.select("id").distinct().orderBy("id").toPandas()["id"].to_list()
    node_id_map = batch_map_ids(node_ids)

    # convert dict back to a dataframe to parallelize the mapping
    node_id_map_df = pd.DataFrame(
        list(node_id_map.items()), columns=["id", "normalized_id"]
    )
    spark = ps.sql.SparkSession.builder.getOrCreate()
    mapping_df = spark.createDataFrame(node_id_map_df)

    # add normalized_id to nodes
    nodes = (
        nodes.join(mapping_df, on="id", how="left")
        .withColumnsRenamed({"id": "original_id"})
        .withColumnsRenamed({"normalized_id": "id"})
    )

    # edges are bit more complex, we need to map both the subject and object
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {"id": "subject", "normalized_id": "subject_normalized"}
        ),
        on="subject",
        how="left",
    )
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {"id": "object", "normalized_id": "object_normalized"}
        ),
        on="object",
        how="left",
    )
    edges = edges.withColumnsRenamed(
        {"subject": "original_subject", "object": "original_object"}
    ).withColumnsRenamed(
        {"subject_normalized": "subject", "object_normalized": "object"}
    )

    return nodes, edges


@retry(stop_after_attempt(3), wait_random_exponential(min=1, max=10))
@memory.cache
def hit_node_norm_service(
    curies: List[str],
    endpoint: str,
    conflate: bool = False,
    drug_chemical_conflate: bool = False,
):
    """Hits the node normalization service with a list of curies.

    Makes heavy use of tenacity for retries and joblib for caching locally on disk

    Args:
        curies (List[str]): A list of curies to normalize.
        endpoint (str): The endpoint to hit.
        conflate (bool, optional): Whether to conflate the nodes. Defaults to False.
        drug_chemical_conflate (bool, optional): Whether to conflate the drug and chemical nodes. Defaults to False.

    Returns:
        Dict[str, str]: A dictionary of the form {id: normalized_id}.

    """
    request_json = {
        "curies": curies,
        "conflate": conflate,
        "drug_chemical_conflate": drug_chemical_conflate,
        "description": "true",
    }

    logger.debug(request_json)
    resp: requests.models.Response = requests.post(url=endpoint, json=request_json)
    logger.debug(resp.json())

    if resp.status_code == 200:
        # if successful return the json as an object
        return _extract_ids(resp.json())
    else:
        logger.warning(f"Node norm response code: {resp.status_code}")
        logger.debug(resp.text)
        resp.raise_for_status()


def _extract_ids(response: Dict[str, Any]):
    ids = {}
    for k in response:
        try:
            if response[k] is None:
                ids[k] = None
            else:
                ids[k] = response[k]["id"]["identifier"]
        except KeyError:
            logger.warning(f"KeyError for {k}: {response[k]}")
            ids[k] = None
    return ids

    return [v["id"]["identifier"] for v in resp.values()]


def batch_map_ids(ids: List[str], batch_size: int = 100, parallelism: int = 10):
    """Maps a list of ids to their normalized form using the node normalization service.

    Args:
        ids (List[str]): A list of ids to map.
        batch_size (int, optional): The number of ids to map in a single batch. Defaults to 100.
        parallelism (int, optional): The number of threads to use for parallel processing. Defaults to 100.

    Returns:
        Dict[str, str]: A dictionary of the form {id: normalized_id}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    mappings = {}
    total_batches = (len(ids) + batch_size - 1) // batch_size

    def _process_batch(batch):
        return hit_node_norm_service(batch, nn_endpoint)

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            futures.append(executor.submit(_process_batch, batch))

        for future in tqdm(
            as_completed(futures), total=total_batches, desc="Batch Mapping IDs"
        ):
            mappings.update(future.result())

    assert len(mappings) == len(ids)
    return mappings
