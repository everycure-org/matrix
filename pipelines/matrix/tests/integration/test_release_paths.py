import re
from typing import Dict

import pytest
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from matrix.pipeline_registry import register_pipelines
from matrix_gcp_datasets.gcp import LazySparkDataset


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
    """Test that all paths after release nodes are namespaced in a runs subfolder.

    Given:
        A node in the pipeline is after the release nodes (create_prm_unified_nodes, create_prm_unified_edges)

    When:
        The node's output dataset is a LazySparkDataset and not a temporary/cache path

    Then:
        The dataset's path should contain "/runs" in its path
    """
    # Given
    pattern = r"releases/[^/]+/runs"
    pipeline = pipelines["__default__"]
    release_nodes = ["create_prm_unified_nodes", "create_prm_unified_edges"]

    # When
    post_release_pipeline = pipeline.from_nodes(*release_nodes)
    post_release_outputs = post_release_pipeline.only_nodes(
        *[x.name for x in post_release_pipeline.nodes if x.name not in release_nodes]
    ).all_outputs()

    # Then
    for dataset_name in post_release_outputs:
        dataset = catalog._get_dataset(dataset_name)
        if isinstance(dataset, LazySparkDataset) and not any(x in dataset._full_url for x in ["/cache/", "/tmp/"]):
            assert re.search(pattern, dataset._full_url), f"Path {dataset._full_url} does not match pattern {pattern}"
