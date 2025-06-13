import glob
import os
import re
from pathlib import Path

import pytest
import yaml
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.project import configure_project, pipelines

_ALLOWED_LAYERS = [
    "raw",
    "int",
    "prm",
    "feat",
    "model_input",
    "models",
    "model_output",
    "reporting",
    "cache",
]


@pytest.fixture(scope="session", autouse=True)
def _configure_matrix_project() -> None:
    """Configure the project for testing."""
    configure_project("matrix")


@pytest.fixture(autouse=True, scope="session")
def openai_api_env():
    os.environ["OPENAI_API_KEY"] = "foo"


def _pipeline_datasets(pipeline) -> set[str]:
    """Helper function to retrieve all datasets used by a pipeline."""
    return set.union(*[set(node.inputs + node.outputs) for node in pipeline.nodes])


@pytest.mark.skip()
@pytest.mark.parametrize("kedro_context", ["cloud_kedro_context", "base_kedro_context"])
def test_no_parameter_entries_from_catalog_unused(
    kedro_context: KedroContext,
    request: pytest.FixtureRequest,
) -> None:
    """Tests whether all parameter entries from the catalog are used in the pipeline."""
    kedro_context = request.getfixturevalue(kedro_context)
    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_params = [entry for entry in used_conf_entries if "params:" in entry]

    declared_conf_entries = kedro_context.catalog.list()

    declared_params = [
        entry
        for entry in declared_conf_entries
        if "params:" in entry and not ("_overrides" in entry or entry.startswith("params:_"))
    ]

    unused_params = [
        declared_param
        for declared_param in declared_params
        if not any(
            declared_param.startswith(used_param) or used_param.startswith(declared_param) for used_param in used_params
        )
    ]

    # # # Only catalog entry not used should be 'parameters', since we input top-level keys
    # # directly.
    # assert (
    #     unused_data_sets == set()
    # ), f"The following data sets are not used: {unused_data_sets}"
    unused_params_str = "\n".join(unused_params)
    assert unused_params_str == "", f"The following parameters are not used: {unused_params_str}"


@pytest.mark.skip()
@pytest.mark.parametrize("kedro_context", ["cloud_kedro_context", "base_kedro_context"])
def test_no_non_parameter_entries_from_catalog_unused(
    kedro_context: KedroContext,
    request: pytest.FixtureRequest,
) -> None:
    kedro_context = request.getfixturevalue(kedro_context)
    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_entries = {entry for entry in used_conf_entries if "params:" not in entry}
    declared_entries = {entry for entry in kedro_context.catalog.list() if not entry.startswith("params:")}

    unused_entries = declared_entries - used_entries
    # Only catalog entry not used should be 'parameters', since we input top-level keys directly.
    unused_entries.remove("parameters")

    assert unused_entries == set(), f"The following entries are not used: {unused_entries}"


@pytest.mark.integration
def test_memory_data_sets_absent(cloud_kedro_context: KedroContext) -> None:
    """Tests no MemoryDataSets are created."""

    used_data_sets = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_data_sets_wout_double_params = {x.replace("params:params:", "params:") for x in used_data_sets}

    catalog_datasets = set(cloud_kedro_context.catalog.list())
    memory_data_sets = []

    for dataset in used_data_sets_wout_double_params:
        if dataset not in catalog_datasets:
            # Check if the dataset can be resolved by factory patterns
            try:
                # This internally handles pattern matching and dataset creation
                cloud_kedro_context.catalog._get_dataset(dataset)
            except Exception:
                # If it can't be resolved, it would be a MemoryDataset
                memory_data_sets.append(dataset)

    assert len(memory_data_sets) == 0, f"{memory_data_sets}"


@pytest.mark.integration
@pytest.mark.parametrize("config_loader", ["cloud_config_loader", "base_config_loader"])
def test_catalog_filepath_follows_conventions(
    conf_source: Path, config_loader: OmegaConfigLoader, request: pytest.FixtureRequest
) -> None:
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

    config_loader = request.getfixturevalue(config_loader)

    # Check catalog entries
    failed_results = []

    for pattern in config_loader.config_patterns["catalog"]:
        for file in glob.glob(f"{conf_source}/{pattern}", recursive=True):
            # Load catalog entries
            with open(file) as f:
                entries = yaml.safe_load(f)

            # Extract pipeline name from filepath
            _, pipeline, _ = os.path.relpath(file, conf_source).split(os.sep, 2)

            # Validate each entry
            for entry in entries:
                # Ignore tmp. entries
                if entry.startswith("_"):
                    continue

                expected_pattern = rf"{pipeline}\.({{.*}}\.)*[{' | '.join(_ALLOWED_LAYERS)}]\.*"
                if not re.search(expected_pattern, entry):
                    failed_results.append(
                        {
                            "entry": entry,
                            "expected_pattern": expected_pattern,
                            "filepath": file,
                            "description": f"Expected {expected_pattern} to match filepath.",
                        }
                    )

    assert failed_results == [], f"Entries that failed conventions: {failed_results}"


@pytest.mark.integration
@pytest.mark.parametrize("config_loader", ["cloud_config_loader", "base_config_loader"])
def test_parameters_filepath_follows_conventions(
    conf_source: Path, config_loader: OmegaConfigLoader, request: pytest.FixtureRequest
) -> None:
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

    config_loader = request.getfixturevalue(config_loader)

    # Check catalog entries
    failed_results = []
    for pattern in config_loader.config_patterns["parameters"]:
        for file in glob.glob(f"{conf_source}/{pattern}", recursive=True):
            if os.path.isdir(file):
                continue

            # Load catalog entries
            with open(file) as f:
                entries = yaml.safe_load(f)

            # Extract pipeline name from filepath
            _, pipeline, _ = os.path.relpath(file, conf_source).split(os.sep, 2)

            # Do not allow empty files
            if entries is None:
                failed_results.append(
                    {
                        "filepath": file,
                        "description": "Empty parameters file.",
                    }
                )
                continue

            # Validate each entry
            for entry in entries:
                # Ignore tmp. entries
                if entry.startswith("_"):
                    continue

                expected_pattern = rf"{pipeline}\.*"
                if not re.search(expected_pattern, entry):
                    failed_results.append(
                        {
                            "entry": entry,
                            "expected_pattern": expected_pattern,
                            "filepath": file,
                            "description": f"Expected {expected_pattern} to match filepath.",
                        }
                    )

    assert failed_results == [], f"Entries that failed conventions: {failed_results}"
