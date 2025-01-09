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
                    # This node works on the nodes file, but we also need it for the edge file
                    # We cannot use the same set of controls for edges because we do not need the category_value_count
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
                        name=f"{source['name']}_ingestion_metrics",
                        tags=["qc"],
                    ),
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

    # return pipeline(
    #     [
    #         # RTX-KG2 INGESTION QC
    #         node(
    #             func=nodes.ingestion,
    #             inputs={
    #                 "nodes": "ingestion.int.rtx_kg2.nodes",
    #                 "edges": "ingestion.int.rtx_kg2.edges",
    #                 "dataset_name": "params:dataset_name_rtx_kg2",
    #                 "output_path": "params:output_path_rtx_kg2_ingestion",
    #             },
    #             outputs=None,  # "qc.int.rtx_kg2.qc_results_ingestion",
    #             name="rtx_kg2_ingestion_metrics",
    #             tags=["qc"],
    #         ),
    #         # ROBOKOP INGESTION QC
    #         node(
    #             func=nodes.ingestion,
    #             inputs={
    #                 "nodes": "ingestion.int.robokop.nodes",
    #                 "edges": "ingestion.int.robokop.edges",
    #                 "dataset_name": "params:dataset_name_robokop",
    #                 "output_path": "params:output_path_robokop_ingestion",
    #             },
    #             outputs=None,  # "qc.int.robokop.qc_results_ingestion",
    #             name="robokop_ingestion_metrics",
    #             tags=["qc"],
    #         ),
    #         # RTX-KG2 INTEGRATION QC
    #         node(
    #             func=nodes.integration,
    #             inputs={
    #                 "nodes": "ingestion.int.rtx_kg2.nodes",
    #                 "nodes_transformed": "integration.int.rtx.nodes",
    #                 "norm_nodes": "integration.int.rtx.nodes.norm",
    #                 "norm_nodes_map": "integration.int.rtx.nodes_norm_mapping",
    #                 "dataset_name": "params:dataset_name_rtx_kg2",
    #                 "output_path": "params:output_path_rtx_kg2_integration",
    #             },
    #             outputs=None,  # "qc.int.rtx_kg2.qc_results_integration",
    #             name="rtx_kg2_integration_metrics",
    #             tags=["qc"],
    #         ),
    #         # ROBOKOP INTEGRATION QC
    #         node(
    #             func=nodes.integration,
    #             inputs={
    #                 "nodes": "ingestion.int.robokop.nodes",
    #                 "nodes_transformed": "integration.int.robokop.nodes",
    #                 "norm_nodes": "integration.int.robokop.nodes.norm",
    #                 "norm_nodes_map": "integration.int.robokop.nodes_norm_mapping",
    #                 "dataset_name": "params:dataset_name_robokop",
    #                 "output_path": "params:output_path_robokop_integration",
    #             },
    #             outputs=None,  # "qc.int.robokop.qc_results_integration",
    #             name="robokop_integration_metrics",
    #             tags=["qc"],
    #         ),
    #     ]
    # )
