import logging

from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


# TODO:
# - for each source
#   - monitoring before transformation (need to go through transformer?
#   - monitoring after transformation
#   - after normalization
# - after sources are merged and duplicates removed
# - after filtering
#


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
                        func=nodes.count_untransformed_knowledge_graph,
                        inputs={
                            "transformer": f"params:integration.sources.{source_name}.transformer",
                            "nodes": f"ingestion.int.{source_name}.nodes",
                            "edges": f"ingestion.int.{source_name}.edges",
                        },
                        outputs={
                            "nodes_report": f"integration.reporting.{source_name}_nodes_pre_integration",
                            "edges_report": f"integration.reporting.{source_name}_edges_pre_integration",
                        },
                        name=f"qc_{source_name}_pre_integration",
                    )
                ]
            )
        )
        # Monitor before transformation via specific transformer
        # Monitor after transformation

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.count_filtered_knowledge_graph,
                    inputs={
                        "nodes": "integration.prm.filtered_nodes",
                        "edges": "integration.prm.filtered_edges",
                    },
                    outputs={
                        "nodes_report": "integration.reporting.filtered_knowledge_graph_agg_count",
                        "edges_report": "integration.reporting.filtered_edges_agg_count",
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
            integration_quality_control_pipeline(),
        ]
    )
