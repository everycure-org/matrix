from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # RTX-KG2 INGESTION QC
            node(
                func=nodes.ingestion,
                inputs={
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "edges": "ingestion.int.rtx_kg2.edges",
                    "dataset_name": "ingestion.int.rtx_kg2.nodes",
                },
                outputs="qc.int.rtx_kg2.qc_results",
                name="rtx-kg2_ingestion_metrics",
                tags=["qc"],
            ),
            # ROBOKOP INGESTION QC
            node(
                func=nodes.ingestion,
                inputs={
                    "nodes": "ingestion.int.robokop.nodes",
                    "edges": "ingestion.int.robokop.edges",
                    "dataset_name": "ingestion.int.robokop.nodes",
                },
                outputs="qc.int.robokop.qc_results",
                name="robokop_ingestion_metrics",
                tags=["qc"],
            ),
        ]
    )
