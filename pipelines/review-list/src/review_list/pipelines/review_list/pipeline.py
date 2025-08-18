from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node

from .nodes import combine_ranked_pair_dataframes, prefetch_top_quota


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline with dynamic inputs based on parameters."""

    config_loader = OmegaConfigLoader("conf")
    params = config_loader.get("parameters")
    inputs_config = params.get("inputs_to_review_list", {})

    # Dynamically create dictionary of input datasets
    input_datasets = {}
    for dataset_name, _ in inputs_config.items():
        input_datasets[dataset_name] = dataset_name

    # Dynamically create output names for trimmed datasets
    trimmed_outputs = [f"trimmed_{name}" for name in input_datasets.keys()]

    return Pipeline(
        [
            node(
                func=prefetch_top_quota,
                inputs={
                    **input_datasets,
                    "weights": "params:inputs_to_review_list",
                    "config": "params:review_list_config",
                },
                outputs=trimmed_outputs,
                name="prefetch_top_quota_node",
            ),
            node(
                func=combine_ranked_pair_dataframes,
                inputs={
                    **{name: f"trimmed_{name}" for name in input_datasets.keys()},
                    "weights": "params:inputs_to_review_list",
                    "config": "params:review_list_config",
                },
                outputs="combined_ranked_pair_dataframe",
                name="combine_ranked_pair_dataframes_node",
            ),
        ]
    )
