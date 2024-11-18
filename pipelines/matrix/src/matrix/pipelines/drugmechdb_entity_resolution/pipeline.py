"""Preprocessing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create DrugMechDB entity resolution pipeline."""
    return pipeline(
        [
            node(
                func=nodes.normalize_drugmechdb_entities,
                inputs={
                    "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                    "api_endpoint": "params:integration.nodenorm.api_endpoint",
                    "prenormalize_func": "params:drugmechdb_entity_resolution.prenormalize_func",
                },
                outputs="moa_extraction.raw.drugmechdb_entities",
                name="normalize_drugmechdb_entities",
            ),
        ]
    )
