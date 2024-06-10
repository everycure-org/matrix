import pytest
import glob
import os
import yaml
import re

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


def _pipeline_datasets(pipeline) -> set[str]:
    """Helper function to retrieve all datasets used by a pipeline."""
    return set.union(*[set(node.inputs + node.outputs) for node in pipeline.nodes])


@pytest.mark.integration
def test_unused_catalog_entries(kedro_context, configure_matrix_project):
    """Tests whether all catalog entries are used in the pipeline.

    FUTURE: Fix validating unused dataset entries, this is currently not feasible
    due to the Kedro dataset mechanism.
    """

    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_params = [entry for entry in list(used_conf_entries) if "params:" in entry]

    declared_conf_entries = kedro_context.catalog.list()

    # used_data_sets = {entry for entry in used_conf_entries if "params:" not in entry}
    # declared_data_sets = {entry for entry in declared_conf_entries if "params:" not in entry and entry != "parameters"}

    declared_params = [
        entry
        for entry in declared_conf_entries
        if "params:" in entry
        and not ("_overrides" in entry or entry.startswith("params:_"))
    ]

    unused_params = [
        declared_param
        for declared_param in declared_params
        if not any(
            [
                declared_param.startswith(used_param)
                or used_param.startswith(declared_param)
                for used_param in used_params
            ]
        )
    ]

    # # # Only catalog entry not used should be 'parameters', since we input top-level keys
    # # directly.
    # assert (
    #     unused_data_sets == set()
    # ), f"The following data sets are not used: {unused_data_sets}"

    assert (
        unused_params == []
    ), f"The following parameters are not used: {unused_params}"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ENV_NAME") == "CI", reason="Ongoing issue with the Kedro catalog"
)
def test_memory_data_sets_absent(kedro_context, configure_matrix_project):
    """Tests no MemoryDataSets are created."""

    used_data_sets = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])

    used_data_sets_wout_double_params = {
        x.replace("params:params:", "params:") for x in used_data_sets
    }

    memory_data_sets = {
        dataset
        for dataset in used_data_sets_wout_double_params
        if dataset not in kedro_context.catalog.list()
        and not kedro_context.catalog.exists(dataset)
    }

    assert len(memory_data_sets) == 0, f"{memory_data_sets}"


@pytest.mark.integration
def test_catalog_filepath_follows_conventions(conf_source):
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

    # Check catalog entries
    failed_results = []
    for file in glob.glob(f"{conf_source}/**/*catalog**.y*ml", recursive=True):
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

            expected_pattern = (
                rf"{pipeline}\.({{namespace}}.)?[{' | '.join(_ALLOWED_LAYERS)}]\.*"
            )
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
def test_parameters_filepath_follows_conventions(conf_source):
    """Checks if catalog entry filepaths conform to entry.

    The filepath of the catalog entry should be of the format below. More
    elaborate error checking can be added later.

    Allowed conventions:

        {pipeline}.{layer}.*

        {pipeline}.{namespace}.{layer}.*
    """

    # Check catalog entries
    failed_results = []
    for file in glob.glob(f"{conf_source}/**/*parameters**.y*ml", recursive=True):
        # Load catalog entries
        with open(file) as f:
            entries = yaml.safe_load(f)

        # breakpoint()

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
