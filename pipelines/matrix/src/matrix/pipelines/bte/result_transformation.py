"""This module transforms raw BTE query results into a DataFrame."""

import logging
from typing import List, Dict, Any
import pandas as pd
from pydantic import ValidationError
from .bte_pydantic import ApiResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def transform_results(raw_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transforms raw query results into a DataFrame with columns `target`, `source`, and `score`.

    :param raw_results: List of raw results from the asynchronous queries.
    :return: DataFrame with columns `target`, `source`, and `score`.
    """
    transformed_data = []

    for raw_result in raw_results:
        if raw_result:
            curie = raw_result.get("curie")
            try:
                response = ApiResponse(**raw_result)
            except ValidationError as e:
                logging.error(
                    f"Validation error in transforming results for curie {curie}: {e.errors()}"
                )
                continue

            if response.message and response.message.results:
                for result in response.message.results:
                    n1_node_bindings = result.node_bindings.get("n1", [])
                    if not n1_node_bindings:
                        continue

                    n1_node_id = n1_node_bindings[0].id

                    for n0_binding in result.node_bindings.get("n0", []):
                        chemical_curie = n0_binding.id
                        for analysis in result.analyses or []:
                            print(f"{n1_node_id}: {chemical_curie}: {analysis.score}")
                            transformed_data.append(
                                {
                                    "target": n1_node_id,
                                    "source": chemical_curie,
                                    "score": analysis.score,
                                }
                            )

    result_df = pd.DataFrame(transformed_data, columns=["target", "source", "score"])
    print(result_df.head())
    logging.info(f"Transform Results: DataFrame shape {result_df.shape}")
    return result_df
