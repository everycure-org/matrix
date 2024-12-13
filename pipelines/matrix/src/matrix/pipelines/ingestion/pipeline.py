from kedro.pipeline import Pipeline, pipeline, node

from matrix import settings


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes = []

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

        if source.get("normalize_edges", True):
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
