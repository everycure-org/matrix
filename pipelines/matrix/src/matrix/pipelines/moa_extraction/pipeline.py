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
                inputs={
                    "runner": "params:moa_extraction.neo4j_runner",
                    "drug_types": "params:moa_extraction.preprocessing_options.add_tags.drug_types",
                    "disease_types": "params:moa_extraction.preprocessing_options.add_tags.disease_types",
                    "batch_size": "params:moa_extraction.preprocessing_options.add_tags.batch_size",
                    "verbose": "params:moa_extraction.preprocessing_options.add_tags.verbose",
                },
                outputs=None,
                name="add_tags",
            ),
            node(
                func=nodes.get_one_hot_encodings,
                inputs={"runner": "params:moa_extraction.neo4j_runner"},
                outputs=["moa_extraction.category_encoder", "moa_extraction.relation_encoder"],
                name="get_one_hot_encodings",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline()
