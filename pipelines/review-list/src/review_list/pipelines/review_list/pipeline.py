from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node

from .nodes import combine_ranked_pair_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline with dynamic inputs based on parameters."""

    config_loader = OmegaConfigLoader("conf")
    params = config_loader.get("parameters")
    inputs_config = params.get("inputs_to_review_list", {})

    # Dynamically create dictionary of input datasets and weights
    input_datasets = {}
    weights = {}

    for dataset_name, config in inputs_config.items():
        input_datasets[dataset_name] = dataset_name
        weights[dataset_name] = config.get("weight")

    return Pipeline(
        [
            node(
                func=combine_ranked_pair_dataframes,
                inputs={
                    **input_datasets,
                    "weights": "params:inputs_to_review_list",
                    "config": "params:review_list_config",
                },
                outputs="combined_ranked_pair_dataframe",
                name="combine_ranked_pair_dataframes_node",
            ),
        ]
    )
