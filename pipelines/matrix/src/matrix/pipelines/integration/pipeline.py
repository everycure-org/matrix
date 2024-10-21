"""Integration pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from .robokop import transform_robo_edges, transform_robo_nodes
from .rtxkg2 import transform_rtxkg2_edges, transform_rtxkg2_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=transform_robo_nodes,
                inputs="ingestion.int.robokop.nodes",
                outputs="integration.int.robokop.nodes",
                name="transform_robokop_nodes",
                tags=["standardize"],
            ),
            node(
                func=transform_robo_edges,
                inputs="ingestion.int.robokop.edges",
                outputs="integration.int.robokop.edges",
                name="transform_robokop_edges",
                tags=["standardize"],
            ),
            node(
                func=transform_rtxkg2_nodes,
                inputs="ingestion.int.rtx_kg2.nodes",
                outputs="integration.int.rtx.nodes",
                name="transform_rtx_nodes",
                tags=["standardize"],
            ),
            node(
                func=transform_rtxkg2_edges,
                inputs=["ingestion.int.rtx_kg2.nodes", "integration.raw.biolink.predicates"],
                outputs="integration.int.rtx.edges",
                name="transform_rtx_edges",
                tags=["standardize"],
            ),
            # Normalize the KG IDs
            node(
                func=nodes.normalize_kg,
                inputs={
                    "nodes": "integration.int.rtx.nodes",
                    "edges": "integration.int.rtx.edges",
                    "api_endpoint": "params:integration.nodenorm.api_endpoint",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs=[
                    "integration.int.rtx.nodes.norm",
                    "integration.int.rtx.edges.norm",
                    "integration.int.rtx.nodes_norm_mapping",
                ],
                name="normalize_rtx_kg",
                tags=["standardize"],
            ),
            node(
                func=nodes.normalize_kg,
                inputs={
                    "nodes": "integration.int.robokop.nodes",
                    "edges": "integration.int.robokop.edges",
                    "api_endpoint": "params:integration.nodenorm.api_endpoint",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs=[
                    "integration.int.robokop.nodes.norm",
                    "integration.int.robokop.edges.norm",
                    "integration.int.robokop.nodes_norm_mapping",
                ],
                name="normalize_robokop_kg",
            ),
            node(
                func=nodes.union_nodes,
                inputs={
                    "datasets_to_union": "params:integration.unification.datasets_to_union",
                    "rtx": "integration.int.rtx.nodes.norm",
                    "robokop": "integration.int.robokop.nodes.norm",
                    "medical_team": "ingestion.int.ec_medical_team.nodes",
                },
                outputs="integration.prm.unified_nodes",
                name="create_prm_unified_nodes",
            ),
            # union edges
            node(
                func=nodes.union_edges,
                inputs={
                    "datasets_to_union": "params:integration.unification.datasets_to_union",
                    "rtx": "integration.int.rtx.edges.norm",
                    "robokop": "integration.int.robokop.edges.norm",
                    "medical_team": "ingestion.int.ec_medical_team.edges",
                },
                outputs="integration.prm.unified_edges",
                name="create_prm_unified_edges",
            ),
        ]
    )
