from kedro.pipeline import Pipeline, pipeline, node

from matrix import settings

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create QC pipeline."""
    pipelines = []
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        pipelines.append(
            pipeline(
                [
                    node(
                        func=nodes.run_quality_control,
                        inputs={
                            "df": f"ingestion.int.{source['name']}.nodes",
                            "controls": "params:quality_control.ingestion_nodes",
                        },
                        outputs=f"quality_control.{source['name']}.prm.ingestion_metrics_nodes",
                        name=f"{source['name']}_ingestion_metrics",
                        tags=["qc"],
                    ),
                    # Trying this out and need to confirm with Laurens
                    node(
                        func=nodes.run_quality_control,
                        inputs={
                            "df": f"ingestion.int.{source['name']}.edges",
                            "controls": "params:quality_control.ingestion_edges",
                        },
                        outputs=f"quality_control.{source['name']}.prm.ingestion_metrics_edges",
                        # TODO: Names have to be unique as well
                        name=f"{source['name']}_ingestion_metrics",
                        tags=["qc"],
                    ),
                    # Trying this out and need to confirm with Laurens
                    node(
                        func=nodes.run_quality_control,
                        inputs={
                            "df": f"integration.int.{source['name']}.nodes.norm",
                            "controls": "params:quality_control.normalized_nodes",
                        },
                        outputs=f"quality_control.{source['name']}.prm.normalization_metrics_nodes",
                        name=f"{source['name']}_normalization_metrics",
                        tags=["qc"],
                    ),
                    # Trying this out and need to confirm with Laurens
                    node(
                        func=nodes.run_quality_control,
                        inputs={
                            "df": f"integration.int.{source['name']}.edges.norm",
                            "controls": "params:quality_control.normalized_edges",
                        },
                        outputs=f"quality_control.{source['name']}.prm.normalization_metrics_edges",
                        name=f"{source['name']}_normalization_metrics",
                        tags=["qc"],
                    ),
                    # TODO: add nodes for transformation, unified, filtered steps.
                    # Do we need this anymore?
                    node(
                        func=nodes.run_quality_control,
                        inputs={
                            "df": f"integration.int.{source['name']}.nodes_norm_mapping",
                            "controls": "params:quality_control.integration",
                        },
                        outputs=f"quality_control.{source['name']}.prm.integration_metrics",
                        name=f"{source['name']}_integration_metrics",
                        tags=["qc"],
                    ),
                ],
                tags=[source["name"]],
            )
        )

    return sum(pipelines)
