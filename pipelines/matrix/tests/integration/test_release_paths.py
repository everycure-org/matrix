import re
from typing import Dict

import pytest
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from matrix.datasets.gcp import LazySparkDataset, SparkDatasetWithBQExternalTable
from matrix.pipeline_registry import register_pipelines


@pytest.fixture
def catalog(kedro_session) -> DataCatalog:
    """Load the complete catalog configuration from the cloud environment using Kedro's session management.

    Args:
        kedro_session: Kedro session fixture that handles project configuration and Spark session.

    Returns:
        DataCatalog: A Kedro DataCatalog instance with all cloud environment datasets.
    """
    return kedro_session.load_context().catalog


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
    """

    print("Catalog contents:", catalog._datasets)

    pattern = r"releases/[^/]+/runs"  # Removed the square brackets as they were causing issues

    # All pipelines after the release
    pipeline = pipelines["__default__"]
    release_nodes = ["create_prm_unified_nodes", "create_prm_unified_edges"]
    from_nodes_pipeline = pipeline.from_nodes(*release_nodes)

    outputs = from_nodes_pipeline.only_nodes(
        *[x.name for x in from_nodes_pipeline.nodes if x.name not in release_nodes]
    ).all_outputs()

    for dataset_name in outputs:
        print(dataset_name)
        dataset = catalog._get_dataset(dataset_name)
        print(dataset)

        if isinstance(dataset, (LazySparkDataset, SparkDatasetWithBQExternalTable)):
            print("is instance of LazySparkDataset or SparkDatasetWithBQExternalTable")
            print(dataset._filepath)
            print(dataset._full_url)
            assert re.search(
                pattern, str(dataset._filepath)
            ), f"Path {dataset._filepath} does not match pattern {pattern}"
        else:
            print("is not instance of LazySparkDataset or SparkDatasetWithBQExternalTable")

    assert False
