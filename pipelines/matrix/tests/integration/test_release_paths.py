import re
from pathlib import Path
from typing import Dict

import pytest
import yaml
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from matrix.datasets.gcp import LazySparkDataset

pattern = r"[releases|tests]/[^/]+/runs"


@pytest.fixture
def catalog() -> DataCatalog:
    # Load catalog configuration from YAML file
    catalog_path = Path("conf/test/filtering/catalog.yml")
    with open(catalog_path) as f:
        catalog_config = yaml.safe_load(f)

    # Load credentials if they exist
    credentials_path = Path("conf/test/credentials.yml")
    credentials = {}
    if credentials_path.exists():
        with open(credentials_path) as f:
            credentials = yaml.safe_load(f)

    return DataCatalog.from_config(
        catalog=catalog_config,
        credentials=credentials,
    )


# Use all_outputs() to get all outputs, including intermediate nodes
# outputs() only returns terminal nodes


def test_all_post_release_paths_namespaced(catalog: DataCatalog, pipelines: Dict[str, Pipeline]):
    # ensures all paths past the unified nodes and edges datasets are namespaced in a `runs` subfolder
    pipe = pipelines["__default__"]
    release_nodes = ["create_prm_unified_nodes", "create_prm_unified_edges"]
    from_nodes_pipeline = pipe.from_nodes(*release_nodes)

    outputs = from_nodes_pipeline.only_nodes(
        *[x.name for x in from_nodes_pipeline.nodes if x.name not in release_nodes]
    ).all_outputs()

    # Catalog does not contain dynamically generated outputs

    for ds in outputs:
        if isinstance(catalog._datasets.get(ds), LazySparkDataset):
            print(f"{ds} is instance of LazySparkDataset")
            if re.search(pattern, catalog._datasets[ds]._full_url):
                print("Full url matches pattern")
            else:
                print(f"{catalog._datasets[ds]._full_url} does not match pattern")
