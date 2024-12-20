from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create QC pipeline."""
    return pipeline(
        [
            # RTX-KG2 INGESTION QC
            node(
                func=nodes.ingestion,
                inputs={
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "edges": "ingestion.int.rtx_kg2.edges",
                    "dataset_name": "params:dataset_name_rtx_kg2",
                    "output_path": "params:output_path_rtx_kg2_ingestion",
                },
                outputs=None,  # "qc.int.rtx_kg2.qc_results_ingestion",
                name="rtx_kg2_ingestion_metrics",
                tags=["qc"],
            ),
            # ROBOKOP INGESTION QC
            node(
                func=nodes.ingestion,
                inputs={
                    "nodes": "ingestion.int.robokop.nodes",
                    "edges": "ingestion.int.robokop.edges",
                    "dataset_name": "params:dataset_name_robokop",
                    "output_path": "params:output_path_robokop_ingestion",
                },
                outputs=None,  # "qc.int.robokop.qc_results_ingestion",
                name="robokop_ingestion_metrics",
                tags=["qc"],
            ),
            # RTX-KG2 INTEGRATION QC
            node(
                func=nodes.integration,
                inputs={
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "nodes_transformed": "integration.int.rtx.nodes",
                    "norm_nodes": "integration.int.rtx.nodes.norm",
                    "norm_nodes_map": "integration.int.rtx.nodes_norm_mapping",
                    "dataset_name": "params:dataset_name_rtx_kg2",
                    "output_path": "params:output_path_rtx_kg2_integration",
                },
                outputs=None,  # "qc.int.rtx_kg2.qc_results_integration",
                name="rtx_kg2_integration_metrics",
                tags=["qc"],
            ),
            # ROBOKOP INTEGRATION QC
            node(
                func=nodes.integration,
                inputs={
                    "nodes": "ingestion.int.robokop.nodes",
                    "nodes_transformed": "integration.int.robokop.nodes",
                    "norm_nodes": "integration.int.robokop.nodes.norm",
                    "norm_nodes_map": "integration.int.robokop.nodes_norm_mapping",
                    "dataset_name": "params:dataset_name_robokop",
                    "output_path": "params:output_path_robokop_integration",
                },
                outputs=None,  # "qc.int.robokop.qc_results_integration",
                name="robokop_integration_metrics",
                tags=["qc"],
            ),
        ]
    )
