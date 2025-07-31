from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode
from matrix.pipelines.data_release import last_node_name

# Last node is made explicit because there's a kedro hook after_node_run
# being triggered after the completion of the last node of this pipeline.
# This node is monitored by the data release workflow for successful completion.
# It's a sentinel indicating all data-delivering nodes are really done executing.
# It _must_ be the very last node in this pipeline.


def get_sentinel_inputs(is_patch: bool, full_e2e_run: bool) -> list[str]:
    kg_release_patch_outputs = ["data_release.prm.kgx_edges", "data_release.prm.kgx_nodes"] + [
        f"integration.int.{source['name']}.normalization_summary"
        for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
    ]
    kg_release_outputs = kg_release_patch_outputs + ["data_release.prm.kg_edges", "data_release.prm.kg_nodes"]

    if full_e2e_run:
        # Can add other modelling pipeline outputs.
        return ["matrix_transformations.full_matrix_output@spark"]

    # We might be able to remove this if we run the modelling pipeline every time we do a patch.
    elif is_patch:
        return kg_release_patch_outputs
    else:
        return kg_release_outputs


def sentinel_function(*args):
    return True


def create_pipeline(is_patch: bool = False, full_e2e_run: bool = False, **kwargs) -> Pipeline:
    sentinel_inputs = get_sentinel_inputs(is_patch, full_e2e_run)
    return pipeline(
        [
            ArgoNode(
                func=sentinel_function,
                inputs=sentinel_inputs,
                outputs="data_release.dummy",
                name=last_node_name,
            )
        ]
    )
