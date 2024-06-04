"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Write nodes
            node(
                func=nodes.extract_nodes,
                inputs=["integration.raw.rtx_kg2.nodes", "params:modelling.drug_types"],
                outputs="integration.prm.drugs",
                name="create_neo4j_drug_nodes",
            ),
            node(
                func=nodes.extract_nodes,
                inputs=[
                    "integration.raw.rtx_kg2.nodes",
                    "params:modelling.disease_types",
                ],
                outputs="integration.prm.diseases",
                name="create_neo4j_disease_nodes",
            ),
            # Write relationship
            node(
                func=nodes.extract_edges,
                inputs=["integration.raw.rtx_kg2.edges"],
                outputs="integration.prm.treats",
                name="create_neo4j_edges",
            ),
            # Example reading
            node(
                func=nodes.neo4j_decorated,
                inputs=["integration.prm.pypher", "params:modelling.drug_types"],
                outputs=None,
                name="print",
            ),
        ]
    )
