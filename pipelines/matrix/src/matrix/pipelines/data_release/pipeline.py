from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from matrix.pipelines.data_release import last_node_name
from matrix.pipelines.data_release.nodes import unified_edges_to_kgx, unified_nodes_to_kgx

# Last node is made explicit because there's a kedro hook after_node_run
# being triggered after the completion of the last node of this pipeline.
# This node is monitored by the data release workflow for successful completion.
# It's a sentinel indicating all data-delivering nodes are really done executing.
# It _must_ be the very last node in this pipeline.
last_node = ArgoNode(
    func=lambda x, y, z: True,
    inputs=["data_release.prm.kgx_edges", "data_release.prm.kgx_nodes", "data_release.prm.kg_edges"],
    outputs="data_release.dummy",
    name=last_node_name,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            ArgoNode(
                func=unified_edges_to_kgx,
                inputs=["integration.prm.unified_edges"],
                outputs="data_release.prm.kgx_edges",
                name="write_edges_to_kgx",
                tags=["kgx"],
            ),
            ArgoNode(
                func=unified_nodes_to_kgx,
                inputs=["integration.prm.unified_nodes"],
                outputs="data_release.prm.kgx_nodes",
                name="write_nodes_to_kgx",
                tags=["kgx"],
            ),
            last_node,
            # NOTE: Enable when the embeddings pipeline worked prior to this pipeline
            # # release to neo4j
            # ArgoNode(
            #     func=lambda x: x,
            #     inputs=["embeddings.feat.nodes"],
            #     outputs="data_release.feat.nodes_with_embeddings",
            #     name="ingest_nodes_with_embeddings",
            # ),
            # NOTE: Enable if you want embeddings
            # ArgoNode(
            #     func=lambda _, x: x,
            #     inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="data_release.prm.kg_embeddings",
            # )
            # need nodes that bring the nodes/edges to BigQuery
        ]
    )
