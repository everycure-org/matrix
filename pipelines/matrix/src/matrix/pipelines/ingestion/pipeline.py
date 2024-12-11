from kedro.pipeline import Pipeline, pipeline, node

from matrix import settings


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    pipelines = []

    # Add ingestion pipeline for each source
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        for component in ["nodes", "edges"]:
            pipelines.append(
                pipeline(
                    [
                        node(
                            func=lambda x: x,
                            inputs=[f'ingestion.raw.{source["name"]}.{component}@spark'],
                            outputs=f'ingestion.int.{source["name"]}.{component}',
                            name=f'write_{source["name"]}_{component}',
                            tags=[f'{source["name"]}'],
                        )
                    ]
                )
            )

    return sum(pipelines)
