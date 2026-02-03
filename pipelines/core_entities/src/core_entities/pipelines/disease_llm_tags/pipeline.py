from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_llm_preprocessing_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.ingest_source_disease_list,
                inputs="raw.disease_list",
                outputs="primary.disease_list",
                name="ingest_source_disease_list",
            ),
            node(
                func=nodes.ingest_curated_disease_list,
                inputs="raw.curated_disease_list",
                outputs="primary.curated_disease_list",
                name="ingest_curated_disease_list",
            ),
            node(
                func=nodes.merge_disease_lists,
                inputs={
                    "disease_list": "primary.disease_list",
                    "curated_disease_list": "primary.curated_disease_list",
                },
                outputs="primary.disease_llm_list",
                name="merge_disease_lists",
            ),
            node(
                lambda x: x,
                inputs="input.disease_name_patch",
                outputs="primary.disease_name_patch",
                name="ingest_disease_name_patch",
            ),
            node(
                func=nodes.patch_disease_name,
                inputs={
                    "disease_list": "primary.disease_llm_list",
                    "disease_name_patch": "primary.disease_name_patch",
                },
                outputs="primary.disease_list_patched",
                name="patch_disease_name",
            ),
        ]
    )


def create_disease_categories_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_llm_preprocessing_pipeline(),
            node(
                func=nodes.invoke_graph,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "graph": "params:disease_categories.graph",
                    "invoke_parameters": "params:disease_categories.invoke_parameters",
                    "parallelism": "params:disease_categories.parallelism",
                    "ignore_errors": "params:ignore_llm_pipeline_errors",
                },
                name="get_disease_categories",
                outputs="primary.release.disease_categories",
            ),
        ]
    )


def create_disease_labels_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_llm_preprocessing_pipeline(),
            node(
                func=nodes.invoke_graph,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "graph": "params:disease_labels.graph",
                    "invoke_parameters": "params:disease_labels.invoke_parameters",
                    "parallelism": "params:disease_labels.parallelism",
                    "ignore_errors": "params:ignore_llm_pipeline_errors",
                },
                name="get_disease_labels",
                outputs="primary.release.disease_labels",
            ),
        ]
    )


def create_disease_umn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_llm_preprocessing_pipeline(),
            node(
                func=nodes.invoke_graph,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "graph": "params:disease_umn.graph",
                    "invoke_parameters": "params:disease_umn.invoke_parameters",
                    "parallelism": "params:disease_umn.parallelism",
                    "ignore_errors": "params:ignore_llm_pipeline_errors",
                },
                name="get_disease_umn",
                outputs="primary.release.disease_umn",
            ),
        ]
    )


def create_disease_prevalence_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_llm_preprocessing_pipeline(),
            node(
                func=nodes.ingest_disease_labels,
                inputs="raw.disease_labels",
                outputs="primary.disease_labels",
                name="ingest_disease_labels",
            ),
            node(
                func=nodes.merge_disease_list_with_labels,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "disease_labels": "primary.disease_labels",
                },
                outputs="primary.disease_list_patched_with_labels",
                name="merge_disease_list_with_labels",
            ),
            node(
                func=nodes.invoke_graph,
                inputs={
                    "disease_list": "primary.disease_list_patched_with_labels",
                    "graph": "params:disease_prevalence.graph",
                    "invoke_parameters": "params:disease_prevalence.invoke_parameters",
                    "parallelism": "params:disease_prevalence.parallelism",
                    "ignore_errors": "params:ignore_llm_pipeline_errors",
                },
                name="get_disease_prevalence",
                outputs="primary.release.disease_prevalence",
            ),
        ]
    )


def create_disease_txgnn_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            create_llm_preprocessing_pipeline(),
            node(
                func=nodes.invoke_graph,
                inputs={
                    "disease_list": "primary.disease_list_patched",
                    "graph": "params:disease_txgnn.graph",
                    "invoke_parameters": "params:disease_txgnn.invoke_parameters",
                    "parallelism": "params:disease_txgnn.parallelism",
                    "ignore_errors": "params:ignore_llm_pipeline_errors",
                },
                name="get_disease_txgnn",
                outputs="primary.release.disease_txgnn",
            ),
        ]
    )
