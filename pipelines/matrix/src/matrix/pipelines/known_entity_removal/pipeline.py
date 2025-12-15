from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ArgoNode

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    available_datasets = settings.DYNAMIC_PIPELINES_MAPPING().get("known_entity_removal").get("available_datasets")

    return pipeline(
        [
            ArgoNode(
                func=nodes.concatenate_datasets,
                inputs={
                    "datasets_to_include": "params:known_entity_removal.datasets_to_include",
                    **{f"{dataset}": f"integration.int.{dataset}.edges.norm@spark" for dataset in available_datasets},
                },
                name="concatenate_datasets",
                outputs="known_entity_removal.int.concatenated_ground_truth",
            ),
            ArgoNode(
                func=nodes.apply_mondo_expansion,
                inputs={
                    "mondo_ontology": "params:known_entity_removal.mondo_ontology",
                    "concatenated_ground_truth": "known_entity_removal.int.concatenated_ground_truth",
                },
                name="apply_mondo_expansion",
                outputs="known_entity_removal.int.expanded_ground_truth",
            ),
            ArgoNode(
                func=nodes.create_known_entity_matrix,
                inputs={
                    "drug_list": "integration.int.drug_list.nodes.norm@spark",
                    "disease_list": "integration.int.disease_list.nodes.norm@spark",
                    "expanded_ground_truth": "known_entity_removal.int.expanded_ground_truth",
                },
                name="create_known_entity_matrix",
                outputs="known_entity_removal.model_output.known_entity_matrix",
            ),
            ArgoNode(
                func=nodes.preprocess_orchard_pairs,
                inputs={
                    "orchard_pairs": "known_entity_removal.raw.orchard_pairs_by_month",
                },
                name="preprocess_orchard_pairs",
                outputs="known_entity_removal.int.preprocessed_orchard_pairs",
            ),
            ArgoNode(
                func=nodes.restrict_to_report_date,
                inputs={
                    "orchard_pairs": "known_entity_removal.int.preprocessed_orchard_pairs",
                    "orchard_report_date": "params:known_entity_removal.orchard_report_date",
                },
                name="restrict_to_report_date",
                outputs={
                    "restricted_orchard_pairs": "known_entity_removal.model_output.restricted_orchard_pairs",
                    "report_date_info": "known_entity_removal.evaluation.orchard_data_report_date",
                },
            ),
        ]
    )
