import pytest

from kedro.framework.project import pipelines


@pytest.mark.integration
def test_unused_catalog_entries(kedro_context, configure_matrix_project):
    """Tests whether all catalog entries are used in the pipeline."""
    used_conf_entries = set.union(
        *(set(n.inputs + n.outputs) for p in pipelines.values() for n in p.nodes)
    )
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
