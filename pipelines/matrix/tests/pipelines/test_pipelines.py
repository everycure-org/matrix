import pytest

from kedro.framework.project import pipelines


def _pipeline_datasets(pipeline) -> set[str]:
    """Helper function to retrieve all datasets used by a pipeline."""
    return set.union(*[set(node.inputs + node.outputs) for node in pipeline.nodes])


@pytest.mark.integration
def test_unused_catalog_entries(kedro_context, configure_matrix_project):
    """Tests whether all catalog entries are used in the pipeline."""
    used_conf_entries = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])
    used_data_sets = {entry for entry in used_conf_entries if "params:" not in entry}
    used_params = [entry for entry in list(used_conf_entries) if "params:" in entry]

    declared_conf_entries = kedro_context.catalog.list()
    declared_data_sets = {
        entry
        for entry in declared_conf_entries
        if "params:" not in entry and entry != "parameters"
    }
    declared_params = [entry for entry in declared_conf_entries if "params:" in entry]

    unused_data_sets = declared_data_sets - used_data_sets
    unused_params = [
        declared_param
        for declared_param in declared_params
        if not any(  # pylint: disable=use-a-generator
            [declared_param.startswith(used_param) for used_param in used_params]
        )
    ]

    # Only catalog entry not used should be 'parameters', since we input top-level keys
    # directly.
    assert (
        unused_data_sets == set()
    ), f"The following data sets are not used: {unused_data_sets}"

    assert (
        unused_params == []
    ), f"The following parameters are not used: {unused_params}"


@pytest.mark.integration
def test_memory_data_sets_absent(kedro_context, configure_matrix_project):
    """Tests no MemoryDataSets are created."""
    used_data_sets = set.union(*[_pipeline_datasets(p) for p in pipelines.values()])

    used_data_sets_wout_double_params = {
        x.replace("params:params:", "params:") for x in used_data_sets
    }

    memory_data_sets = used_data_sets_wout_double_params - set(
        kedro_context.catalog.list()
    )

    assert len(memory_data_sets) == 0, f"{memory_data_sets}"
