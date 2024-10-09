"""
MOA extraction pipeline.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from . import nodes


def _preprocessing_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.add_tags,
                inputs=[
                    "params:moa_extraction.neo4j_runner",
                    "params:moa_extraction.drug_types",
                    "params:moa_extraction.disease_types",
                ],
                outputs=None,
                name="add_tags",
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline()
