from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from . import nodes


def _create_source_parsing_pipeline(
    source: str,
    source_type: str = "external_registry",
    has_mapping: bool = False,
) -> Pipeline:
    """Create parsing pipeline for a PKS source.

    Args:
        source: Source name (e.g., 'infores', 'reusabledata').
        source_type: Either 'matrix_curated' or 'external_registry'.
        has_mapping: Whether source requires ID mapping to infores.
    """
    pipelines = []

    if source_type == "matrix_curated":
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.parse_pks_source,
                        inputs={
                            "parser": f"params:document_kg.pks_parsing.sources.{source}.parser",
                            "source_data": f"document_kg.raw.{source}_pks@pandas",
                        },
                        outputs=f"document_kg.int.{source}_metadata",
                        name=f"parse_{source}",
                        tags=[
                            "argowf.fuse",
                            "argowf.fuse-group.document_kg",
                        ],
                    ),
                ],
            )
        )
    else:
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.parse_pks_source,
                        inputs={
                            "parser": f"params:document_kg.pks_parsing.sources.{source}.parser",
                            "source_data": f"document_kg.raw.{source}",
                            **({"mapping_data": f"document_kg.raw.mapping_{source}_infores"} if has_mapping else {}),
                        },
                        outputs=f"document_kg.int.{source}_metadata",
                        name=f"parse_{source}",
                        tags=[
                            "argowf.fuse",
                            "argowf.fuse-group.document_kg",
                        ],
                    ),
                ],
            )
        )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Create PKS documentation pipeline."""
    pipelines = []

    for source in settings.DYNAMIC_PIPELINES_MAPPING()["document_kg"]:
        pipelines.append(
            pipeline(
                _create_source_parsing_pipeline(
                    source=source["name"],
                    source_type=source.get("source_type", "external_registry"),
                    has_mapping=source.get("has_mapping", False),
                ),
            )
        )

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.merge_all_pks_metadata,
                    inputs=[
                        *[
                            f"document_kg.int.{source['name']}_metadata"
                            for source in settings.DYNAMIC_PIPELINES_MAPPING()["document_kg"]
                        ]
                    ],
                    outputs="document_kg.int.all_pks_metadata",
                    name="merge_all_pks_metadata",
                    tags=[
                        "argowf.fuse",
                        "argowf.fuse-group.document_kg",
                    ],
                ),
                ArgoNode(
                    func=nodes.integrate_all_metadata,
                    inputs=[
                        "document_kg.int.all_pks_metadata",
                        "integration.prm.unified_edges",
                    ],
                    outputs="document_kg.prm.pks_yaml",
                    name="filter_to_relevant_pks",
                    tags=[
                        "argowf.fuse",
                        "argowf.fuse-group.document_kg",
                    ],
                ),
                ArgoNode(
                    func=nodes.create_pks_documentation,
                    inputs=["document_kg.prm.pks_yaml", "params:document_kg.pks_parsing.templates"],
                    outputs="document_kg.prm.pks_md",
                    name="create_pks_documentation",
                    tags=[
                        "argowf.fuse",
                        "argowf.fuse-group.document_kg",
                    ],
                ),
            ]
        ),
    )

    return sum(pipelines)
