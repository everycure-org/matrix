import pytest
import re

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


# @pytest.mark.integration
# def test_catalog_filepath_follows_conventions(config_loader):
#     """Checks if catalog entry filepaths conform to entry.

#     The filepath of the catalog entry must match the naming of the catalog entry, i.e.:
#     <connector>.<multi>.<sub-multi>.catalog.<name> must match
#     `filepath: .../<connector>/.../<multi>/<sub-multi>/`.

#     Note that catalog and `<name>` is omitted from the search. This means all previous
#     words separated by `.` must be found in the catalog filepath in that specific order.

#     Other examples:
#       - catalog: predictive_modeling.growth.catalog.df
#         expected filepath: bucket/.../predictive_modeling/growth/...
#       - catalog: comm_core.germany.primary_care.catalog.df
#         expected filepath: bucket/.../comm_core/germany/primary_care/...
#     """
#     inferred_flat_conf = config_loader.get("catalog")

#     keys = [
#         {"catalog_entry": key, "filepath": value["filepath"]}
#         for key, value in inferred_flat_conf.items()
#         if "filepath" in value
#     ]

#     parsed_keys = [x["catalog_entry"].split(".")[:-1] for x in keys]

#     failed_results = []
#     for key, pattern in zip(keys, parsed_keys):
#         print(key, pattern)
