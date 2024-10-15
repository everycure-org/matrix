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
                tags="moa_extraction.preprocessing",
                name="add_tags",
            ),
            node(
                func=nodes.get_one_hot_encodings,
                inputs={"runner": "params:moa_extraction.neo4j_runner"},
                outputs=["moa_extraction.feat.category_encoder", "moa_extraction.feat.relation_encoder"],
                name="get_one_hot_encodings",
                tags="moa_extraction.preprocessing",
            ),
            node(
                func=nodes.map_drug_mech_db,
                inputs={
                    "runner": "params:moa_extraction.neo4j_runner",
                    "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                    "mapper": "params:moa_extraction.path_mapping.mapper_two_hop",
                    "synonymizer_endpoint": "params:moa_extraction.path_mapping.synonymizer_endpoint",
                },
                outputs="moa_extraction.int.two_hop_indication_paths",
                name="map_two_hop",
                tags="moa_extraction.preprocessing",
            ),
            node(
                func=nodes.map_drug_mech_db,
                inputs={
                    "runner": "params:moa_extraction.neo4j_runner",
                    "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                    "mapper": "params:moa_extraction.path_mapping.mapper_three_hop",
                    "synonymizer_endpoint": "params:moa_extraction.path_mapping.synonymizer_endpoint",
                },
                outputs="moa_extraction.int.three_hop_indication_paths",
                name="map_three_hop",
                tags="moa_extraction.preprocessing",
            ),  # TODO: Add mapping success report node
            # node(
            #     func=lambda x: breakpoint(),
            #     inputs=["moa_extraction.int.three_hop_indication_paths"],
            #     outputs=None,
            #     name="inspect_three_hop",
            #     tags="moa_extraction.preprocessing",
            # ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline()
