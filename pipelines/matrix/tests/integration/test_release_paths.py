import re
from typing import Dict

import pytest
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from matrix.datasets.gcp import LazySparkDataset
from matrix.pipeline_registry import register_pipelines


@pytest.fixture
def catalog(cloud_kedro_context) -> DataCatalog:
    """Load the complete catalog configuration from the cloud environment.

    Args:
        cloud_kedro_context: Kedro context configured for the cloud environment.

    Returns:
        DataCatalog: A Kedro DataCatalog instance with all cloud environment datasets.
    """
    return cloud_kedro_context.catalog


@pytest.fixture
def pipelines() -> Dict[str, Pipeline]:
    """Get all registered pipelines.

    Returns:
        Dict[str, Pipeline]: Dictionary mapping pipeline names to Pipeline objects.
    """
    return register_pipelines()


def test_all_post_release_paths_namespaced(catalog: DataCatalog, pipelines: Dict[str, Pipeline]):
    """
    Ensures all paths past the unified nodes and edges datasets are namespaced in a `runs` subfolder.
    Excludes temporary and cache paths from this check.
    """
    pattern = r"releases/[^/]+/runs"

    # All pipelines after the release
    pipeline = pipelines["__default__"]
    release_nodes = ["create_prm_unified_nodes", "create_prm_unified_edges"]
    from_nodes_pipeline = pipeline.from_nodes(*release_nodes)

    outputs = from_nodes_pipeline.only_nodes(
        *[x.name for x in from_nodes_pipeline.nodes if x.name not in release_nodes]
    ).all_outputs()

    for dataset_name in outputs:
        dataset = catalog._get_dataset(dataset_name)
        if isinstance(dataset, LazySparkDataset):
            # These two had errors:
            # filepath=gs://mtrx-us-central1-hub-dev-storage/kedro/data/cache/tmp/cache_misses/node_embeddings
            # filepath=/Users/aford/matrix/pipelines/matrix/data/tmp/feat/tmp_nodes_with_embeddings
            if not any(x in dataset._full_url for x in ["/cache/", "/tmp/"]):
                assert re.search(
                    pattern, dataset._full_url
                ), f"Path {dataset._full_url} does not match pattern {pattern}"
