import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes = []

    # Add shared nodes
    nodes.append(
        node(
            func=lambda x: x,
            inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
            outputs="ingestion.int.rtx_kg2.curie_to_pmids",
            name="write_rtx_kg2_curie_to_pmids",
            tags=["rtx_kg2"],
        )
    )

    # Add ingestion pipeline for each source
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        nodes.append(
            node(
                func=lambda x: x,
                inputs=[f'ingestion.raw.{source["name"]}.nodes@spark'],
                outputs=f'ingestion.int.{source["name"]}.nodes',
                name=f'write_{source["name"]}_nodes',
                tags=[f'{source["name"]}'],
            )
        )

        if not source.get("nodes_only", False):
            nodes.append(
                node(
                    func=lambda x: x,
                    inputs=[f'ingestion.raw.{source["name"]}.edges@spark'],
                    outputs=f'ingestion.int.{source["name"]}.edges',
                    name=f'write_{source["name"]}_edges',
                    tags=[f'{source["name"]}'],
                )
            )

    return pipeline(nodes)
