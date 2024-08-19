"""Ingestion pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    return pipeline(
        [
            node(
                func=nodes.upload_to_bq,
                inputs=["ingestion.raw.robokop_kg.nodes@pandas", "ingestion.raw.robokop_kg.edges@pandas"],
                outputs=["ingestion.raw.robokop_kg.nodes@gbq", "ingestion.raw.robokop_kg.edges@gbq"],
                name="normalize_kg_data",
                tags=["RobokopKG"],
            ),
            node(
                func=nodes.normalize_kg_data,
                inputs=["ingestion.raw.robokop_kg.nodes@gbq", "ingestion.raw.robokop_kg.edges@gbq"],
                outputs=["ingestion.int.robokop_kg.nodes@gbq", "ingestion.int.robokop_kg.edges@gbq"],
                name="normalize_kg_data",
                tags=["RobokopKG", "NodeNormalizer"],
            ),
            node(
                func=nodes.upload_to_bq,
                inputs=["ingestion.raw.rtx_kg2.nodes@pandas", "ingestion.raw.rtx_kg2.edges@pandas"],
                outputs=["ingestion.raw.rtx_kg2.nodes@gbq", "ingestion.raw.rtx_kg2.edges@gbq"],
                name="normalize_kg_data",
                tags=["RTX_KG2"],
            ),
            node(
                func=nodes.normalize_kg_data,
                inputs=["ingestion.raw.rtx_kg2.nodes@gbq", "ingestion.raw.rtx_kg2.edges@gbq"],
                outputs=["ingestion.int.rtx_kg2.nodes@gbq", "ingestion.int.rtx_kg2.edges@gbq"],
                name="normalize_kg_data",
                tags=["RTX_KG2", "NodeNormalizer"],
            ),
        ]
    )
