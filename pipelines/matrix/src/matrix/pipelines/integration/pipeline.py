"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from refit.v1.core.inline_has_schema import has_schema


def _create_int_pairs(raw_tp: pd.DataFrame, raw_tn: pd.DataFrame):
    raw_tp["y"] = 1
    raw_tn["y"] = 0

    # Concat
    return pd.concat([raw_tp, raw_tn], axis="index").reset_index(drop=True)


@has_schema(
    schema={
        "label": "string",
        "id": "string",
        "name": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
    },
    allow_subset=True,
)
def _create_nodes(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("label", F.split(F.col("category"), ":", limit=2).getItem(1))
        .withColumn(
            "properties",
            F.create_map(F.lit("foo"), F.lit("bar"), F.lit("bar"), F.lit("foo")),
        )
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )


@has_schema(
    schema={
        "label": "string",
        "source_id": "string",
        "target_id": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
    },
    allow_subset=True,
)
def _create_pairs_neo(df: DataFrame):
    return (
        df.withColumn(
            "label", F.when(F.col("y") == 1, "THREATS").otherwise("NOT_THREATS")
        )
        .withColumn(
            "properties",
            F.create_map(F.lit("foo"), F.lit("bar"), F.lit("bar"), F.lit("foo")),
        )
        .withColumn("source_id", F.col("source"))
        .withColumn("target_id", F.col("target"))
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )


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
            # Write Neo4J nodes
            node(
                func=_create_nodes,
                inputs=["integration.prm.rtx_kg2.nodes"],
                outputs="integration.model_input.nodes",
                name="create_neo4j_nodes",
                tags=["rtx_kg2"],
            ),
            # Construct ground_truth
            node(
                func=_create_int_pairs,
                inputs=[
                    "integration.raw.ground_truth.tp",
                    "integration.raw.ground_truth.tn",
                ],
                outputs="integration.int.known_pairs@pandas",
                name="create_int_known_pairs",
            ),
            node(
                func=_create_pairs_neo,
                inputs=[
                    "integration.int.known_pairs@spark",
                ],
                outputs="integration.model_input.treats",
                name="create_neo4j_known_pairs",
            ),
            # NOTE: Dataset is corrupt
            # node(
            #     func=lambda x: x,
            #     inputs=["integration.raw.rtx_kg2.edges"],
            #     outputs="integration.prm.rtx_kg2.edges",
            #     name="write_rtx_kg2_edges",
            #     tags=["rtx_kg2"],
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
