from kedro.pipeline import Pipeline, node

from review_list.settings import REVIEW_LIST_INPUTS

from .nodes import prefetch_top_quota, weighted_interleave_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline with dynamic inputs based on settings configuration."""

    input_datasets = REVIEW_LIST_INPUTS
    input_datasets_dict = {name: name for name in input_datasets}

    return Pipeline(
        [
            node(
                func=prefetch_top_quota,
                inputs={
                    **input_datasets_dict,
                    "weights": "params:weighted_inputs",
                    "config": "params:review_list_config",
                },
                outputs=[f"trimmed_{name}@spark" for name in input_datasets],
                name="prefetch_top_quota_node",
            ),
            node(
                func=weighted_interleave_dataframes,
                inputs={
                    **{name: f"trimmed_{name}@pandas" for name in input_datasets},
                    "weights": "params:weighted_inputs",
                    "config": "params:review_list_config",
                },
                outputs="combined_ranked_pairs_dataframe@pandas",
                name="weighted_interleave_dataframes_node",
            ),
        ]
    )
