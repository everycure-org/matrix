from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # Bucketize and partition nodes
            ArgoNode(
                func=nodes.bucketize_df,
                inputs={
                    "df": "modelling.prm.filtered_nodes",
                    "input_features": "params:embeddings.node.input_features",
                    "bucket_size": "params:embeddings.node.batch_size",
                    "max_input_len": "params:embeddings.node.max_input_len",
                },
                outputs="embeddings.feat.bucketized_nodes@spark",
                name="bucketize_nodes",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Compute embeddings
            ArgoNode(
                func=nodes.compute_embeddings,
                inputs={
                    "dfs": "embeddings.feat.bucketized_nodes@partitioned",
                    "encoder": "params:embeddings.node.encoder",
                },
                outputs="embeddings.feat.graph.node_embeddings@partitioned",
                name="add_node_embeddings",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Reduce dimension
            ArgoNode(
                func=nodes.reduce_embeddings_dimension,
                inputs={
                    "df": "embeddings.feat.graph.node_embeddings@spark",
                    "unpack": "params:embeddings.dimensionality_reduction",
                },
                outputs="embeddings.feat.graph.pca_node_embeddings",
                name="apply_pca",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            ArgoNode(
                func=nodes.filter_edges_for_topological_embeddings,
                inputs=[
                    "modelling.prm.filtered_nodes",
                    "modelling.prm.filtered_edges",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="embeddings.feat.graph.edges_for_topological",
                name="filter_edges_for_topological",
            ),
            # Load spark dataset into local neo instance
            ArgoNode(
                # NOTE: We are only selecting these two categories due to OOM neo4j instance error
                # which occurs when we select all cols
                func=lambda x: x.select("id", "pca_embedding"),
                inputs=["embeddings.feat.graph.pca_node_embeddings"],
                outputs="embeddings.tmp.input_nodes",
                name="ingest_neo4j_input_nodes",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            ArgoNode(
                func=nodes.ingest_edges,
                inputs=[
                    "embeddings.tmp.input_nodes",
                    "embeddings.feat.graph.edges_for_topological",
                ],
                outputs="embeddings.tmp.input_edges",
                name="ingest_neo4j_input_edges",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
                # FUTURE: Ensure we define "packages / tshirt size standard configurations" for resources
                argo_config=ArgoResourceConfig(
                    cpu_request=48,
                    cpu_limit=48,
                    memory_limit=192,
                    memory_request=120,
                ),
            ),
            ArgoNode(
                func=nodes.train_topological_embeddings,
                inputs={
                    "df": "embeddings.tmp.input_edges",
                    "gds": "params:embeddings.gds",
                    "topological_estimator": "params:embeddings.topological_estimator",
                    "unpack": "params:embeddings.topological",
                },
                outputs=["embeddings.models.topological", "embeddings.reporting.loss"],
                name="train_topological_embeddings",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            ArgoNode(
                func=nodes.write_topological_embeddings,
                inputs={
                    "model": "embeddings.models.topological",
                    "gds": "params:embeddings.gds",
                    "topological_estimator": "params:embeddings.topological_estimator",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.model_output.topological",
                name="add_topological_embeddings",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.mem-100g",
                    "argowf.template-neo4j",
                ],
            ),
            # extracts the nodes from neo4j
            ArgoNode(
                func=nodes.extract_topological_embeddings,
                inputs={
                    "embeddings": "embeddings.model_output.topological",
                    "nodes": "modelling.prm.filtered_nodes",
                    "string_col": "params:embeddings.write_topological_col",
                },
                outputs="embeddings.feat.nodes",
                name="extract_nodes_edges_from_db",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            # Create PCA plot
            ArgoNode(
                func=nodes.reduce_dimension,
                inputs={
                    "df": "embeddings.feat.nodes",
                    "unpack": "params:embeddings.topological_pca",
                },
                outputs="embeddings.reporting.topological_pca",
                name="apply_topological_pca",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_pca",
                ],
            ),
            ArgoNode(
                func=nodes.visualise_pca,
                inputs={
                    "nodes": "embeddings.reporting.topological_pca",
                    "column_name": "params:embeddings.topological_pca.output",
                },
                outputs="embeddings.reporting.topological_pca_plot",
                name="create_pca_plot_topological_embeddings",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_pca",
                ],
            ),
        ],
    )
