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
                outputs="known_entity_removal.concatenated_ground_truth",
            )
        ]
    )
