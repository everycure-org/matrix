from pathlib import Path
import pytest
import glob
import os
import yaml
import re

from kedro.framework.project import configure_project
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.project import pipelines

_ALLOWED_LAYERS = [
    "raw",
    "int",
    "prm",
    "feat",
    "model_input",
    "models",
    "model_output",
    "reporting",
]


@pytest.fixture(scope="session", autouse=True)
def _configure_matrix_project() -> None:
    """Configure the project for testing."""
    configure_project("matrix")


def _pipeline_datasets(pipeline) -> set[str]:
    """Helper function to retrieve all datasets used by a pipeline."""
    return set.union(*[set(node.inputs + node.outputs) for node in pipeline.nodes])


@pytest.fixture(autouse=True, scope="session")
def openai_api_env():
    os.environ["OPENAI_API_KEY"] = "foo"


def test_no_catalog_param_entries_unused(
    kedro_context: KedroContext,
) -> None:
    """Tests whether all catalog param entries are used in the pipeline.

    FUTURE: Fix validating unused dataset entries, this is currently not feasible
    due to the Kedro dataset mechanism.
    """

    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_params = [entry for entry in list(used_conf_entries) if "params:" in entry]

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
            [
                declared_param.startswith(used_param) or used_param.startswith(declared_param)
                for used_param in used_params
            ]
        )
    ]

    assert unused_params == [], f"The following parameters are not used: {unused_params}"


def test_no_catalog_non_param_entries_unused(
    kedro_context: KedroContext,
) -> None:
    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_entries = {entry for entry in used_conf_entries if "params:" not in entry}
    declared_entries = {entry for entry in kedro_context.catalog.list() if not entry.startswith("params:")}

    unused_entries = declared_entries - used_entries

    assert unused_entries == set(), f"The following entries are not used: {unused_entries}"


@pytest.mark.integration
# @pytest.mark.skipif(
#     os.environ.get("CI") == "true", reason="Ongoing issue with the Kedro catalog"
# )
# skipping due to dynamic pipelines not being supported at the moment
@pytest.mark.skip()
def test_memory_data_sets_absent(kedro_context: KedroContext) -> None:
    """Tests no MemoryDataSets are created."""

    used_data_sets = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])

    used_data_sets_wout_double_params = {x.replace("params:params:", "params:") for x in used_data_sets}

    memory_data_sets = {
        dataset
        for dataset in used_data_sets_wout_double_params
        if dataset not in kedro_context.catalog.list() and not kedro_context.catalog.exists(dataset)
    }

    assert len(memory_data_sets) == 0, f"{memory_data_sets}"


@pytest.mark.integration
def test_catalog_filepath_follows_conventions(conf_source: Path, config_loader: OmegaConfigLoader) -> None:
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

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
            for entry, _ in entries.items():
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
def test_parameters_filepath_follows_conventions(conf_source, config_loader):
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

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

            # Validate each entry
            for entry, _ in entries.items():
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
