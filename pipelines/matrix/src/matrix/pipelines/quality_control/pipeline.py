import logging

from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def ingestion_quality_control_pipeline() -> Pipeline:
    pipelines = []

    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        if not source["integrate_in_kg"]:
            continue

        source_name = source["name"]
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.count_untransformed_knowledge_graph,
                        inputs={
                            "transformer": f"params:integration.sources.{source_name}.transformer",
                            "nodes": f"ingestion.int.{source_name}.nodes",
                            "edges": f"ingestion.int.{source_name}.edges",
                        },
                        # TODO: create catalog entries
                        outputs={
                            "nodes_report": f"ingestion.reporting.{source_name}_nodes",
                            "edges_report": f"ingestion.reporting.{source_name}_edges",
                        },
                        name=f"qc_{source_name}_ingested",
                    )
                ]
            )
        )

    return sum(pipelines)


def integration_quality_control_pipeline() -> Pipeline:
    pipelines = []

    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        if not source["integrate_in_kg"]:
            continue

        source_name = source["name"]
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.count_knowledge_graph,
                        inputs={
                            "nodes": f"integration.int.{source_name}.nodes",
                            "edges": f"integration.int.{source_name}.edges",
                        },
                        # TODO: create catalog entries
                        outputs={
                            "nodes_report": f"integration.reporting.{source_name}_nodes_transformed",
                            "edges_report": f"integration.reporting.{source_name}_edges_transformed",
                        },
                        name=f"qc_{source_name}_transformed",
                    ),
                    ArgoNode(
                        func=nodes.count_knowledge_graph,
                        inputs={
                            "nodes": f"integration.int.{source_name}.nodes.norm@spark",
                            "edges": f"integration.int.{source_name}.edges",
                        },
                        # TODO: create catalog entries
                        outputs={
                            "nodes_report": f"integration.reporting.{source_name}_nodes_normalized",
                            "edges_report": f"integration.reporting.{source_name}_edges_normalized",
                        },
                        name=f"qc_{source_name}_normalized",
                    ),
                ]
            )
        )

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.count_knowledge_graph,
                    inputs={
                        "nodes": "integration.prm.unified_nodes",
                        "edges": "integration.prm.unified_edges",
                    },
                    # TODO: create catalog entries
                    outputs={
                        "nodes_report": "integration.reporting.nodes_unified",
                        "edges_report": "integration.reporting.edges_unified",
                    },
                    name="qc_unified_knowledge_graph",
                ),
                ArgoNode(
                    func=nodes.count_knowledge_graph,
                    inputs={
                        "nodes": "integration.prm.prefiltered_nodes",
                        "edges": "integration.prm.filtered_edges",
                    },
                    # TODO: create catalog entries
                    outputs={
                        "nodes_report": "integration.reporting.prefiltered_nodes_count",
                        "edges_report": "integration.reporting.prefiltered_edges_count",
                    },
                    name="qc_prefiltered_knowledge_graph",
                ),
                ArgoNode(
                    func=nodes.count_knowledge_graph,
                    inputs={
                        "nodes": "integration.prm.filtered_nodes",
                        "edges": "integration.prm.filtered_edges",
                    },
                    outputs={
                        "nodes_report": "integration.reporting.filtered_nodes_count",
                        "edges_report": "integration.reporting.filtered_edges_count",
                    },
                    name="qc_filtered_knowledge_graph",
                ),
            ]
        )
    )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            ingestion_quality_control_pipeline(),
            integration_quality_control_pipeline(),
        ]
    )
