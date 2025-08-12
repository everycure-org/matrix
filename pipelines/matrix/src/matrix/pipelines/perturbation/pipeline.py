"""Perturbation pipeline for edge rewiring experiments."""

from kedro.pipeline import Pipeline, node

from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create perturbation pipeline."""

    pipeline_nodes = [
        ArgoNode(
            func=nodes.perturb_edges_rewire,
            inputs=[
                "filtering.prm.filtered_edges",
                "filtering.prm.filtered_nodes",
                "params:perturbation.rate",
                "params:perturbation.random_seed",
                "params:perturbation.strategy",
                "params:perturbation.enabled",
            ],
            outputs="perturbation.prm.perturbed_edges",
            name="perturb_edges_rewire",
            tags=["argowf.fuse", "argowf.fuse-group.perturbation", "perturbation"],
            argo_config=ArgoResourceConfig(
                memory_request=128,
                memory_limit=128,
                cpu_request=32,
                cpu_limit=32,
            ),
        ),
        node(
            func=nodes.log_rewiring_stats,
            inputs=[
                "filtering.prm.filtered_edges",
                "perturbation.prm.perturbed_edges",
                "params:perturbation.rate",
                "filtering.prm.filtered_nodes",
            ],
            outputs=None,
            name="log_rewiring_statistics",
            tags=["perturbation", "logging"],
        ),
    ]

    return Pipeline(pipeline_nodes)
