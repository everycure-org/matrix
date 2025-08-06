from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode
from matrix.pipelines.data_release import last_node_name


# Last node is made explicit because there's a kedro hook after_node_run
# being triggered after the completion of the last node of this pipeline.
# This node is monitored by the data release workflow for successful completion.
# It's a sentinel indicating all data-delivering nodes are really done executing.
# It _must_ be the very last node in this pipeline.
def get_sentinel_inputs(is_patch: bool) -> list[str]:
    kg_release_patch_outputs = (
        [
            "data_release.prm.kgx_edges",
            "data_release.prm.kgx_nodes",
            "integration.prm.nodes_edges_consistency_check",
        ]
        + [
            f"integration.int.{source['name']}.normalization_summary"
            for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration")
        ]
        # The master evaluation report is the final node
        # Also check for full matrix output as this runs in parallel with the evaluation pipeline and is critical
        + [
            "evaluation.matrix_transformations.reporting.master_report",
            "matrix_transformations.full_matrix_output@spark",
        ]
    )

    kg_release_outputs = kg_release_patch_outputs + ["data_release.prm.kg_edges", "data_release.prm.kg_nodes"]

    if is_patch:
        return kg_release_patch_outputs
    else:
        return kg_release_outputs


def sentinel_function(*args):
    return True


def create_pipeline(is_patch, **kwargs) -> Pipeline:
    sentinel_inputs = get_sentinel_inputs(is_patch)
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
