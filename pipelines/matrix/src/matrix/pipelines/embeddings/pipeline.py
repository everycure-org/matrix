"""Embeddings pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from . import new_nodes as nn


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            node(
                func=nn.filter_missing_embeddings,
                inputs={
                    "data": "integration.prm.unified_nodes",
                    "cache": "embeddings.cache.existing_invocations",
                },
                outputs=[
                    "enrichment.node_embeddings.missing",
                    "enrichment.node_embeddings.existing",
                ],
            ),
            node(
                func=nn.compute_embeddings,
                inputs={
                    "data": "enrichment.node_embeddings.missing",
                    "model": "params:embeddings.ai_config.model",
                },
                outputs="embeddings.node_embeddings.new",
            ),
            node(
                func=nn.combine_frames,
                inputs={
                    "existing": "embeddings.node_embeddings.existing",
                    "new": "embeddings.node_embeddings.new",
                },
                outputs=[
                    "embeddings.node_embeddings.combined"
                    "embeddings.cache.new_invocations"
                ],
            ),
            # # Compute node embeddings
            # node(
            #     func=nodes.compute_embeddings,
            #     inputs={
            #         "input": "integration.prm.unified_nodes",
            #         "features": "params:embeddings.node.features",
            #         "unpack": "params:embeddings.ai_config",
            #     },
            #     outputs="embeddings.feat.graph.node_embeddings",
            #     name="add_node_embeddings",
            # ),
            # # Reduce dimension
            # node(
            #     func=nodes.reduce_dimension,
            #     inputs={
            #         "df": "embeddings.feat.graph.node_embeddings",
            #         "unpack": "params:embeddings.dimensionality_reduction",
            #     },
            #     outputs="embeddings.feat.graph.pca_node_embeddings",
            #     name="apply_pca",
            # ),
            # # Load spark dataset into local neo instance
            # node(
            #     func=lambda x: x.select("id", "name", "category", "pca_embedding"),
            #     inputs=["embeddings.feat.graph.pca_node_embeddings"],
            #     outputs="embeddings.tmp.input_nodes",
            #     name="ingest_neo4j_input_nodes",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # node(
            #     func=nodes.ingest_edges,
            #     inputs=[
            #         "embeddings.tmp.input_nodes",
            #         "integration.prm.unified_edges",
            #     ],
            #     outputs="embeddings.tmp.input_edges",
            #     name="ingest_neo4j_input_edges",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # node(
            #     func=nodes.add_include_in_graphsage,
            #     inputs={
            #         "df": "embeddings.tmp.input_edges",
            #         "gdb": "params:embeddings.gdb",
            #         "drug_types": "params:modelling.drug_types",
            #         "disease_types": "params:modelling.disease_types",
            #     },
            #     outputs="embeddings.feat.include_in_graphsage@yaml",
            #     name="filter_graphsage",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # node(
            #     func=nodes.train_topological_embeddings,
            #     inputs={
            #         "df": "embeddings.feat.include_in_graphsage@yaml",
            #         "gds": "params:embeddings.gds",
            #         "topological_estimator": "params:embeddings.topological_estimator",
            #         "unpack": "params:embeddings.topological",
            #     },
            #     outputs=["embeddings.models.graphsage", "embeddings.reporting.loss"],
            #     name="train_topological_embeddings",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # node(
            #     func=nodes.write_topological_embeddings,
            #     inputs={
            #         "model": "embeddings.models.graphsage",
            #         "gds": "params:embeddings.gds",
            #         "topological_estimator": "params:embeddings.topological_estimator",
            #         "unpack": "params:embeddings.topological",
            #     },
            #     outputs="embeddings.model_output.graphsage",
            #     name="add_topological_embeddings",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.mem-100g",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # # extracts the nodes from neo4j and writes them to BigQuery
            # node(
            #     func=nodes.extract_node_embeddings,
            #     inputs={
            #         "nodes": "embeddings.model_output.graphsage",
            #         "string_col": "params:embeddings.write_topological_col",
            #     },
            #     outputs="embeddings.feat.nodes",
            #     name="extract_nodes_edges_from_db",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_embeddings",
            #         "argowf.template-neo4j",
            #     ],
            # ),
            # # Create PCA plot
            # node(
            #     func=nodes.reduce_dimension,
            #     inputs={
            #         "df": "embeddings.feat.nodes",
            #         "unpack": "params:embeddings.topological_pca",
            #     },
            #     outputs="embeddings.reporting.topological_pca",
            #     name="apply_topological_pca",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_pca",
            #     ],
            # ),
            # node(
            #     func=nodes.visualise_pca,
            #     inputs={
            #         "nodes": "embeddings.reporting.topological_pca",
            #         "column_name": "params:embeddings.topological_pca.output",
            #     },
            #     outputs="embeddings.reporting.topological_pca_plot",
            #     name="create_pca_plot_topological_embeddings",
            #     tags=[
            #         "argowf.fuse",
            #         "argowf.fuse-group.topological_pca",
            #     ],
            # ),
        ],
    )
