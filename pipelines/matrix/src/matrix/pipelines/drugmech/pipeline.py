"""Integration pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame, SparkSession


def parse_gt(gt: DataFrame) -> DataFrame:
    return gt.withColumnRenamed("source", "drug_id").withColumnRenamed("target", "disease_id").limit(100)


def apply_join(tps: DataFrame):
    spark_session = SparkSession.builder.getOrCreate()

    all_pairs = (
        spark_session.read.format("org.neo4j.spark.DataSource")
        .option("database", "moa")
        .option("url", "bolt://127.0.0.1:7687")
        .option("authentication.type", "basic")
        .option("authentication.basic.username", "neo4j")
        .option("authentication.basic.password", "admin")
        .option(
            "query",
            """
            MATCH (source:is_drug), (source:is_drug)-[:is_treats]->(target:is_disease)
            WITH id(source) as sourceId, collect(id(target)) as targets
            CALL apoc.path.expandConfig(sourceId, {
                terminatorNodes: targets,
                relationshipFilter: '>',
                labelFilter: '-Drug-SmallMolecule/is_disease',
                minLevel: 2,
                maxLevel: 3,
                uniqueness: 'NODE_PATH'
            })
            YIELD path
            WITH [n IN nodes(path) | n.id] as p
            RETURN p
            """,
        )
        .option("partitions", 4)
    ).load()

    # result = (
    #     tps
    #     .join(all_pairs, on=["drug_id", "disease_id"], how="left")
    #     .filter(F.col("path").isNotNull())
    # )

    # all_pairs.show()
    # print(all_pairs.show(truncate=False))
    print(all_pairs.count())

    return all_pairs


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(func=parse_gt, inputs=["tps"], outputs="int.tps", name="preprocess_tps", tags=["index"]),
            node(
                func=lambda x: x.select("drug_id").withColumnRenamed("drug_id", "id"),
                inputs=["int.tps"],
                outputs="int.drug.index",
                name="create_input_drug_index",
                tags=["index"],
            ),
            node(
                func=lambda x: x.select("disease_id").withColumnRenamed("disease_id", "id"),
                inputs=["int.tps"],
                outputs="int.disease.index",
                name="create_input_disease_index",
                tags=["index"],
            ),
            node(
                func=lambda x: x.withColumnRenamed("source", "drug_id").withColumnRenamed("target", "disease_id"),
                inputs=["tps"],
                outputs="int.edges_index",
                name="create_edges_disease_index",
                tags=["index"],
            ),
            node(
                func=apply_join,
                inputs=[
                    "int.tps",
                ],
                outputs="output_df",
                name="create_paths",
                tags=["generate"],
            ),
        ]
    )
