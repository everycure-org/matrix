from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings

from . import nodes


def _create_source_parsing_pipeline(
    source: str,
    source_type: str = "external_registry",
    has_mapping: bool = False,
) -> Pipeline:
    """Create parsing pipeline for a single PKS source.

    Args:
        source: Name of the source (e.g., 'infores', 'reusabledata')
        source_type: Type of source ('external_registry' or 'matrix_curated')
        has_mapping: Whether this source requires ID mapping

    Returns:
        Pipeline for parsing this source
    """
    pipelines = []

    if source_type == "matrix_curated":
        pipelines.append(
            pipeline(
                [
                    node(
                        func=nodes.parse_pks_source,
                        inputs={
                            "parser": f"params:document_kg.pks_parsing.sources.{source}.parser",
                            "source_data": f"document_kg.raw.{source}_pks@pandas",
                            "config": f"params:document_kg.pks_parsing.sources.{source}",
                        },
                        outputs=f"document_kg.int.{source}_metadata",
                        name=f"parse_{source}",
                        tags=["document_kg", "parsing"],
                    ),
                ],
                tags=source,
            )
        )
    else:  # external_registry
        pipelines.append(
            pipeline(
                [
                    node(
                        func=nodes.parse_pks_source,
                        inputs={
                            "parser": f"params:document_kg.pks_parsing.sources.{source}.parser",
                            "source_data": f"document_kg.raw.{source}",
                            "config": f"params:document_kg.pks_parsing.sources.{source}",
                            **({"mapping_data": f"document_kg.raw.mapping_{source}_infores"} if has_mapping else {}),
                        },
                        outputs=f"document_kg.int.{source}_metadata",
                        name=f"parse_{source}",
                        tags=["document_kg", "parsing"],
                    ),
                ],
                tags=source,
            )
        )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Generate documentation and metadata as part of the release process pipeline.

    Uses dynamic pipeline generation pattern similar to integration pipeline.
    """
    pipelines = []

    # Create pipeline per source
    for source in settings.DYNAMIC_PIPELINES_MAPPING()["document_kg"]:
        pipelines.append(
            pipeline(
                _create_source_parsing_pipeline(
                    source=source["name"],
                    source_type=source.get("source_type", "external_registry"),
                    has_mapping=source.get("has_mapping", False),
                ),
                tags=[source["name"], "document_kg"],
            )
        )

    # Add merge, filter, and documentation pipeline
    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.merge_all_pks_metadata,
                    inputs=[
                        *[
                            f"document_kg.int.{source['name']}_metadata"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING()["document_kg"]
                        ]
                    ],
                    outputs="document_kg.int.all_pks_metadata",
                    name="merge_all_pks_metadata",
                    tags=["document_kg"],
                ),
                node(
                    func=nodes.integrate_all_metadata,
                    inputs=[
                        "document_kg.int.all_pks_metadata",
                        "integration.prm.unified_edges",
                    ],
                    outputs="document_kg.prm.pks_yaml",
                    name="filter_to_relevant_pks",
                    tags=["document_kg"],
                ),
                node(
                    func=nodes.create_pks_documentation,
                    inputs="document_kg.prm.pks_yaml",
                    outputs="document_kg.prm.pks_md",
                    name="create_pks_documentation",
                    tags=["document_kg", "formatting"],
                ),
            ]
        )
    )

    return sum(pipelines)
