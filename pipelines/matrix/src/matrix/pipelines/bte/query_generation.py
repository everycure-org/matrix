"""This module generates query data for the BTE pipeline."""

import logging
from typing import List, Dict, Any
import pandas as pd
from .bte_pydantic import Query


def remove_empty_fields(d: Any) -> Any:
    """Recursively remove empty fields (None, empty lists, empty strings, empty dicts, sets, tuples)from a dictionary or list.

    :param d: Dictionary or list to clean.
    :return: Cleaned dictionary or list.
    """
    if isinstance(d, dict):
        return {
            k: remove_empty_fields(v)
            for k, v in d.items()
            if v not in [None, [], "", {}, (), set()]
        }
    elif isinstance(d, list):
        return [
            remove_empty_fields(v) for v in d if v not in [None, [], "", {}, (), set()]
        ]
    return d


def generate_trapi_query(curie: str) -> Query:
    """Generate a TRAPI query for a given CURIE.

    :param curie: The CURIE for which to generate the query.
    :return: The generated Query object.
    :raises ValueError: If the CURIE is invalid.
    """
    if not curie:
        logging.error(f"Invalid curie: {curie}")
        raise ValueError("CURIE must not be empty")

    query_data = {
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

    return Query(**query_data)


def generate_queries() -> List[Dict[str, Any]]:
    # disease_list: pd.DataFrame
    """Generate a list of TRAPI queries from a DataFrame of diseases.

    :param disease_list: DataFrame containing the list of diseases.
    :return: List of dictionaries, each containing a CURIE and its corresponding TRAPI query.
    """
    diseases = pd.read_csv(
        "https://github.com/everycure-org/matrix-disease-list/releases/latest/download/matrix-disease-list.tsv",
        sep="\t",
    )
    queries = []
    count = 0
    for index, row in diseases.iterrows():
        count += 1
        if count > 2:
            break
        if row.get("category_class"):
            curie = row["category_class"]
            trapi_query = generate_trapi_query(curie)
            queries.append({"curie": curie, "query": trapi_query, "index": index})
    return queries
