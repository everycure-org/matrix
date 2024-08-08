"""Ingestion pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    return pipeline(
        [
            node(
                func=nodes.normalize_kg_data,
                inputs=["ingestion.raw.robokop_kg.nodes@pandas", "ingestion.raw.robokop_kg.edges@pandas"],
                outputs=["ingestion.int.robokop_kg.nodes@spark", "ingestion.int.robokop_kg.edges@spark"],
                name="normalize_kg_data",
                tags=["RobokopKG", "NodeNormalizer"],
            ),
        ]
    )
