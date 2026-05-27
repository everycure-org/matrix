# NOTE: This file was partially generated using AI assistance.

from kedro.pipeline import Pipeline, node, pipeline


def create_publish_hf_pipeline(**kwargs) -> Pipeline:
    """
    Create pipeline for publishing EC indications list to Hugging Face.

    This pipeline reads the EC indications list from GCS and publishes it
    to the Hugging Face dataset repository.

    Returns:
        A Kedro pipeline for publishing to Hugging Face.
    """
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="raw.ec_indications_list",
                outputs="primary.published.indications_list_hf",
                name="publish_indications_list_hf",
            ),
        ]
    )
