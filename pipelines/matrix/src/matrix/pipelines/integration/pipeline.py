"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Write kg2
            node(
                func=lambda x: x,
                inputs=["integration.raw.rtx_kg2.nodes"],
                outputs="integration.prm.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            # NOTE: Dataset is corrupt
            # node(
            #     func=lambda x: x,
            #     inputs=["integration.raw.rtx_kg2.edges"],
            #     outputs="integration.prm.rtx_kg2.edges",
            #     name="write_rtx_kg2_edges",
            #     tags=["rtx_kg2"],
            # ),
            # # Write nodes
            # node(
            #     func=nodes.extract_nodes,
            #     inputs=["integration.raw.rtx_kg2.nodes", "params:modelling.drug_types"],
            #     outputs="integration.prm.drugs",
            #     name="create_neo4j_drug_nodes",
            #     tags=["neo4j"],
            # ),
            # node(
            #     func=nodes.extract_nodes,
            #     inputs=[
            #         "integration.raw.rtx_kg2.nodes",
            #         "params:modelling.disease_types",
            #     ],
            #     outputs="integration.prm.diseases",
            #     name="create_neo4j_disease_nodes",
            #     tags=["neo4j"],
            # ),
            # # Write relationship
            # node(
            #     func=nodes.extract_edges,
            #     inputs=["integration.raw.rtx_kg2.edges"],
            #     outputs="integration.prm.treats",
            #     name="create_neo4j_edges",
            #     tags=["neo4j"],
            # ),
            # # Example reading
            # node(
            #     func=nodes.neo4j_decorated,
            #     inputs=["integration.prm.pypher", "params:integration.drug_label"],
            #     outputs=None,
            #     name="print",
            #     tags=["neo4j"],
            # ),
            # # Write table
            # node(
            #     func=lambda x: x,
            #     inputs=["integration.raw.rtx_kg2.nodes"],
            #     outputs="integration.raw.bigquery.edges",
            #     name="write_bigquery_data",
            #     tags=["bigquery"],
            # ),
            # node(
            #     func=lambda x: x.show(),
            #     inputs=["integration.raw.bigquery.edges"],
            #     outputs=None,
            #     name="read_bigquery_data",
            #     tags=["bigquery"],
            # ),
        ]
    )
