from kedro.pipeline import Pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes

# Can add runs here
# Or auto generated based on current run
RUNS_TO_COMPARE = [
    {
        "name": "june_2025_t3",
        "storage_path": "gs://path/to/run_1",
    },
    {
        "name": "july_2025_t3",
        "storage_path": "gs://path/to/run_2",
    },
]


def create_pipeline(**kwargs) -> Pipeline:
    """Create cross-run- comparison evaluation pipeline."""
    # First do some matrix cleaning - take overlapping pairs etc
    pipeline_nodes = [
        # Then perform each evaluation comparison in the existing notebook
        ArgoNode(
            func=nodes.recall_at_n_plots,
            inputs={
                # Add generic catalog entries
                "matrices": [x["storage_path"] for x in RUNS_TO_COMPARE],
                "model_names": [x["name"] for x in RUNS_TO_COMPARE],
            },
            outputs=["run_comparison.plots.recall_at_n"],
            name="plot_recall_at_n",
        ),
        # Add recall@n off-label, commonality, entropy, etc
    ]

    return Pipeline(pipeline_nodes)
