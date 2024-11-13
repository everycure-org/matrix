from kedro.pipeline import Pipeline, node, pipeline
from matrix.pipelines.sample import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            # Sample data
            node(
                func=nodes.sample,
                inputs={
                    "gt_positives": "modelling.raw.ground_truth_in.positives@spark",
                    "gt_negatives": "modelling.raw.ground_truth_in.negatives@spark",
                    "nodes": "integration.prm.filtered_nodes_in",
                    "edges": "integration.prm.filtered_edges_in",
                    "unpack": "params:sampling.configuration",
                },
                outputs={
                    "gt_positives": "modelling.raw.ground_truth.positives@spark",
                    "gt_negatives": "modelling.raw.ground_truth.negatives@spark",
                    "nodes": "integration.prm.filtered_nodes",
                    "edges": "integration.prm.filtered_edges",
                },
                name="sample_data",
            ),
            # Sample embeddings for modelling pipeline
            node(
                func=nodes.reduce_embeddings,
                inputs=["embeddings.feat.nodes_in", "integration.prm.filtered_nodes"],
                outputs="embeddings.feat.nodes",
            ),
        ]
    )
