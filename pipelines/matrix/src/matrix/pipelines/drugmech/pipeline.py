"""Integration pipeline."""

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame, SparkSession


def apply_join(gt_pos: pd.DataFrame):
    spark_session = SparkSession.builder.getOrCreate()

    shards = {}

    for index, row in gt_pos.toPandas().iloc[:10].iterrows():
        shards[f"shard={index}"] = lambda: (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", "moa")
            .option("url", "bolt://127.0.0.1:7687")
            .option("authentication.type", "basic")
            .option("authentication.basic.username", "neo4j")
            .option("authentication.basic.password", "admin")
            .option(
                "query",
                f"""
                MATCH p=(drug:Entity {{id: '{row["source"]}'}})-[*2..3]->(disease:Entity {{id: '{row["target"]}'}})
                WITH [node IN nodes(p) | node.id] as nodes, [node in nodes(p) | labels(node)[1]] as labels, [rel in relationships(p) | type(rel)] as rels
                RETURN nodes, labels, rels
            """,
            )
            .load()
        )

    return shards


def load_full(df: DataFrame):
    df.show()


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # node(
            #     func=lambda x: x.select("drug_id").withColumnRenamed("drug_id", "id"),
            #     inputs=["int.tps"],
            #     outputs="int.drug.index",
            #     name="create_input_drug_index",
            #     tags=["index"],
            # ),
            # node(
            #     func=lambda x: x.select("disease_id").withColumnRenamed("disease_id", "id"),
            #     inputs=["int.tps"],
            #     outputs="int.disease.index",
            #     name="create_input_disease_index",
            #     tags=["index"],
            # ),
            # node(
            #     func=lambda x: x.withColumnRenamed("source", "drug_id").withColumnRenamed("target", "disease_id"),
            #     inputs=["tps"],
            #     outputs="int.edges_index",
            #     name="create_edges_disease_index",
            #     tags=["index"],
            # ),
            node(
                func=apply_join,
                inputs=[
                    "modelling.raw.positives",
                ],
                outputs="drugmech.paths@partitioned",
                name="create_paths",
                tags=["generate"],
            ),
            node(
                func=load_full,
                inputs=[
                    "drugmech.paths@spark",
                ],
                outputs=None,
                name="load_paths",
            ),
        ]
    )
