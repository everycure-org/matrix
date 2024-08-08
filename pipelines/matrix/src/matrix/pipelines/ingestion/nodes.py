"""Nodes for the ingestion pipeline."""
from more_itertools import chunked
import pandas as pd

import logging

from matrix.pipelines.ingestion.normalizers import NodeNormalizer

logger = logging.getLogger(__name__)


def normalize_kg_data(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Call NodeNormalizer to correct nodes_df.id, edges_df.subject, edges_df.object.

    Args:
        nodes_df: Nodes
        edges_df: Edges

    Returns:
        Corrected Nodes & Edges DataFrames
    """
    # column_names: list[str] = df.columns.to_list()
    # new_column_names = {c: c.split(":")[0] for c in column_names}
    # df = df.rename(columns=new_column_names)
    id_column_values = nodes_df[["id"]].values.tolist()
    id_column_values_flattened = [','.join(col).strip() for col in id_column_values]
    logger.debug(f"number of entries: {len(id_column_values_flattened)}")
    # logger.info(f"id_column_values_flattened: {id_column_values_flattened}")

    batched_columns = list(chunked(id_column_values_flattened, 1000))
    logger.debug(f"number of batches to normalize: {len(batched_columns)}")
    cached_node_norms: dict = {}
    node_normalizer = NodeNormalizer()
    for column_batch in batched_columns:
        logger.debug(f"number of unique batch entries: {len(column_batch)}")
        nn_json_response = node_normalizer.hit_node_norm_service(curies=column_batch)
        if nn_json_response:
            # merge the normalization results with what we have gotten so far
            cached_node_norms.update(**nn_json_response)
        else:
            # this shouldn't happen but if the API returns an empty dict instead of nulls,
            # assume none of the curies normalize
            empty_responses = {curie: None for curie in column_batch}
            cached_node_norms.update(empty_responses)

    for nn_key, nn_value in cached_node_norms.items():
        if nn_value is None:
            continue
        nodes_df.loc[nodes_df.id == nn_key, 'id'] = nn_value['id']['identifier']
        edges_df.loc[edges_df.subject == nn_key, 'subject'] = nn_value['id']['identifier']
        edges_df.loc[edges_df.object == nn_key, 'object'] = nn_value['id']['identifier']

    return nodes_df, edges_df
