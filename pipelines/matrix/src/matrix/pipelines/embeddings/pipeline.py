from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # Bucketize and partition nodes
            node(
                func=nodes.bucketize_df,
                inputs={
                    "df": "integration.prm.filtered_nodes",
                    "input_features": "params:embeddings.node_buckets.input_features",
                    "bucket_size": "params:embeddings.node_buckets.batch_size",
                    "max_input_len": "params:embeddings.node_buckets.max_input_len",
                },
                outputs="embeddings.feat.bucketized_nodes@spark",
                name="bucketize_nodes",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Compute embeddings
            node(
                func=nodes.compute_embeddings,
                inputs={
                    "dfs": "embeddings.feat.bucketized_nodes@partitioned",
                    "encoder": "params:embeddings.node_encoder",
                },
                outputs="embeddings.feat.graph.node_embeddings@partitioned",
                name="add_node_embeddings",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Reduce dimension
            node(
                func=nodes.reduce_dimension,
                inputs={
                    "df": "embeddings.feat.graph.node_embeddings@spark",
                    "unpack": "params:embeddings.dimensionality_reduction",
                },
                outputs="embeddings.feat.graph.pca_node_embeddings",
                name="apply_pca",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Load spark dataset into local neo instance
            node(
                func=lambda x: x.select("id", "name", "category", "pca_embedding"),
                inputs=["embeddings.feat.graph.pca_node_embeddings"],
                outputs="embeddings.tmp.input_nodes",
                name="ingest_neo4j_input_nodes",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            node(
                func=nodes.ingest_edges,
                inputs=[
                    "embeddings.tmp.input_nodes",
                    "integration.prm.filtered_edges",
                ],
                outputs="embeddings.tmp.input_edges",
                name="ingest_neo4j_input_edges",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            node(
                func=nodes.add_include_in_topological,
                inputs={
                    "df": "embeddings.tmp.input_edges",
                    "gdb": "params:embeddings.gdb",
                    "drug_types": "params:modelling.drug_types",
                    "disease_types": "params:modelling.disease_types",
                },
                outputs="embeddings.feat.include_in_topological@yaml",
                name="filter_topological",
                tags=[
                    "argowf.fuse",
                    "argowf.fuse-group.topological_embeddings",
                    "argowf.template-neo4j",
                ],
            ),
            node(
                func=nodes.train_topological_embeddings,
                inputs={
                    "df": "embeddings.feat.include_in_topological@yaml",
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
            node(
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
            # extracts the nodes from neo4j and writes them to BigQuery
            node(
                func=nodes.extract_node_embeddings,
                inputs={
                    "embeddings": "embeddings.model_output.topological",
                    "nodes": "integration.prm.filtered_nodes",
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
            node(
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
            node(
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
