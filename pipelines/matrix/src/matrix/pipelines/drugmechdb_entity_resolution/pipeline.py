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
                    "name_resolver": "params:preprocessing.translator.name_resolver",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="moa_extraction.raw.drugmechdb_entities",
                name="normalize_drugmechdb_entities",
            ),
        ]
    )
