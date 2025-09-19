from kedro.pipeline import Pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes
from .settings import DYNAMIC_PIPELINE_MAPPING

INPUTS = DYNAMIC_PIPELINE_MAPPING["run_comparison"]["inputs"]


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run- comparison evaluation pipeline."""
    # First do some matrix cleaning - take overlapping pairs etc
    pipeline_nodes = [
        # Then perform each evaluation comparison in the existing notebook
        ArgoNode(
            func=nodes.recall_at_n_plots,
            # NOTE: This node was partially generated using AI assistance.
            inputs=[name for name in INPUTS.keys()],
            outputs=["run_comparison.plots.recall_at_n"],
            name="plot_recall_at_n",
        ),
        # Add recall@n off-label, commonality, entropy, etc
    ]

    return Pipeline(pipeline_nodes)
