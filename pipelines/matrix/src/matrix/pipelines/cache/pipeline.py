from kedro.pipeline import Pipeline, node, pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            ArgoNode(
                func=nodes.cache,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark", "embeddings.cache"],
                outputs=["embeddings.cache_out", "result"],
                name="cache",
            )
        ]
    )
