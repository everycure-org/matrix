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

    for dataset_name, config in inputs_config.items():
        # dataset_name is the base name (e.g., "input_dataframe_1")
        input_datasets[dataset_name] = (
            dataset_name  # Now catalog entries don't have @spark
        )
        weights[dataset_name] = config.get("weight", 1.0)

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
                    "config": "params:review_list_config",  # Add config parameters
                },
                outputs="combined_ranked_pair_dataframe@spark",
                name="combine_ranked_pair_dataframes_node",
            ),
        ]
    )
