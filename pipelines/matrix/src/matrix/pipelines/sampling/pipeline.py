"""Fabricator pipeline."""
from kedro.pipeline import Pipeline, node, pipeline
from ..integration.nodes import unify_nodes, unify_edges
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create fabricator pipeline."""
    return pipeline(
        [
            node(
                func=unify_nodes,
                inputs=[
                    "ingestion.int.rtx_kg2.nodes",
                    "ingestion.int.ec_medical_team.nodes",
                ],
                outputs="integration.tmp.unified_nodes",
                name="create_prm_unified_nodes",
            ),
            # union edges
            node(
                func=unify_edges,
                inputs=[
                    "ingestion.int.rtx_kg2.edges",
                    "ingestion.int.ec_medical_team.edges",
                ],
                outputs="integration.tmp.unified_edges",
                name="create_prm_unified_edges",
            ),
            node(
                func=nodes.sample_nodes,
                inputs=[
                    "integration.tmp.unified_nodes",
                    "params:sampling.stratify_by",
                    "params:sampling.sampling_ratio",
                    "params:sampling.sampling_seed",
                ],
                outputs="integration.prm.unified_nodes",
                name="extract_nodes",
            ),
            node(
                func=nodes.sample_edges,
                inputs=[
                    "integration.tmp.unified_edges",
                    "integration.prm.unified_nodes",
                    "params:sampling.sampling_ratio",
                    "params:sampling.sampling_seed",
                ],
                outputs="integration.prm.unified_edges",
                name="extract_edges",
            ),
        ]
    )
