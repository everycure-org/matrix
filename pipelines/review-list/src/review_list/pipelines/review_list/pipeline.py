# NOTE: This file was partially generated using AI assistance.

from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node

from .nodes import (
    combine_ranked_pair_dataframes,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline with dynamic inputs based on parameters."""

    # Load parameters using OmegaConfigLoader for Kedro 0.19.13
    config_loader = OmegaConfigLoader("conf")
    params = config_loader.get("parameters")
    inputs_config = params.get("inputs_to_review_list", {})

    # Dynamically create dictionary of input datasets and weights
    input_datasets = {}
    weights = {}

    for catalog_entry, config in inputs_config.items():
        # catalog_entry is already the full catalog name (e.g., "input_dataframe_1@spark")
        # Extract the base name for the key (remove @spark suffix)
        base_name = catalog_entry.split("@")[0]
        input_datasets[base_name] = catalog_entry
        weights[base_name] = config.get("weight", 1.0)

    return Pipeline(
        [
            node(
                func=combine_ranked_pair_dataframes,
                # inputs={
                #     "dataframes": list(input_datasets.values()),  # List of catalog entries
                #     "weights": "params:inputs_to_review_list",  # Parameter reference
                # },
                inputs={
                    **input_datasets,  # Unpack dataset catalog names
                    "weights": "params:inputs_to_review_list",  # Add weights as parameters
                },
                outputs="combined_ranked_pair_dataframe@spark",
                name="combine_ranked_pair_dataframes_node",
            ),
        ]
    )
