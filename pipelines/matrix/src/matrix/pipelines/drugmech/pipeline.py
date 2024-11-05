"""Integration pipeline."""

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def apply_join(gt_pos: pd.DataFrame):
    spark_session = SparkSession.builder.getOrCreate()

    shards = {}
    df = gt_pos.groupBy("source").agg(F.collect_list("target").alias("targets"))

    for index, row in df.toPandas().iloc[:100].iterrows():
        targets = ",".join(f'"{target}"' for target in row["targets"])

        shards[f"shard={index}"] = lambda: (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", "moa")
            .option("url", "bolt://127.0.0.1:7687")
            .option("authentication.type", "basic")
            .option("authentication.basic.username", "neo4j")
            .option("authentication.basic.password", "admin")
            #  {{id: '{row["target"]}'}}
            .option(
                "query",
                f"""
                    MATCH p=(drug:Entity {{id: '{row["source"]}'}})-[*2..3]->(disease:Entity) 
                    WHERE disease.id IN [{targets}] 
                    WITH [node IN nodes(p) | node.id] as nodes, [node in nodes(p) | labels(node)[1]] as labels, [rel in relationships(p) | type(rel)] as rels 
                    RETURN nodes, labels, rels""",
            )
            .load()
        )

    return shards


def load_full(df: DataFrame):
    print(df.count())
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
                tags=["generate"],
            ),
        ]
    )
