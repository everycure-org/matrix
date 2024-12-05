import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, pipeline, node

from matrix import settings

import pyspark.sql.types as T
from pyspark.sql import DataFrame


def format_drugmech(paths: DataFrame) -> DataFrame:
    edges = paths.withColumn("path", F.from_json(F.col("path"), T.ArrayType(T.StringType())))

    # Setup edges
    edges = (
        edges.select("index", F.posexplode(F.col("path")).alias("pos", "element"))
        .withColumn("next_pos", F.col("pos") + 1)
        .join(
            edges.select("index", F.posexplode(F.col("path")).alias("next_pos", "next_element")),
            on=["index", "next_pos"],
            how="inner",
        )
        .select("index", F.col("element").alias("subject"), F.col("next_element").alias("object"))
    )

    # Setup nodes
    nodes = edges.select(F.col("subject").alias("id")).unionByName(edges.select(F.col("object").alias("id"))).distinct()

    return {"nodes": nodes, "edges": edges}


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    pipelines = []

    # Extract nodes and edges from paths for drugmech
    pipelines.append(
        pipeline(
            [
                node(
                    func=format_drugmech,
                    inputs=["ingestion.pre.drugmech.paths@spark"],
                    outputs={
                        "nodes": "ingestion.raw.drugmech.nodes@spark",
                        "edges": "ingestion.raw.drugmech.edges@spark",
                    },
                    name="preprocess_drugmech",
                    tags=["drugmech"],
                )
            ]
        )
    )

    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("integration"):
        for component in ["nodes", "edges"]:
            pipelines.append(
                pipeline(
                    [
                        node(
                            func=lambda x: x,
                            inputs=[f'ingestion.raw.{source["name"]}.{component}@spark'],
                            outputs=f'ingestion.int.{source["name"]}.{component}',
                            name=f'write_{source["name"]}_{component}',
                            tags=[f'{source["name"]}'],
                        )
                    ]
                )
            )

    return sum(pipelines)
