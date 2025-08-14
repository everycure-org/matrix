from kedro.pipeline import Pipeline, node

from .nodes import (
    combine_ranked_pair_dataframes,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=combine_ranked_pair_dataframes,
                inputs=[
                    "input_dataframe_1@spark",
                    "input_dataframe_2@spark",
                ],
                outputs="combined_ranked_pair_dataframe@spark",
                name="combine_ranked_pair_dataframes_node",
            ),
        ]
    )
