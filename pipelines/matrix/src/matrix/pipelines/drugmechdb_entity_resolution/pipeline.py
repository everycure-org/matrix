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
                    "synonymizer_endpoint": "params:drugmechdb_entity_resolution.synonymizer_endpoint",
                },
                outputs="moa_extraction.raw.drugmechdb_entities",
                name="normalize_drugmechdb_entities",
            ),
        ]
    )
