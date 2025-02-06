import pyspark.sql as ps
import pyspark.sql.functions as f
import pytest
from matrix.pipelines.create_sample.samplers import GroundTruthRandomSampler

original_knowledge_graph_nodes_count = 500
original_knowledge_graph_edges_count = 1000
ground_truth_edges_count = 25


@pytest.fixture(scope="module")
def original_knowledge_graph_nodes(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_nodes_count).select("id")


@pytest.fixture(scope="module")
def original_knowledge_graph_edges(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_edges_count).select(
        f.col("id").alias("object"), (f.col("id") % original_knowledge_graph_nodes_count).alias("subject")
    )


@pytest.fixture(scope="module")
def ground_truth_edges(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(ground_truth_edges_count).select(
        f.col("id").alias("object"),
        (f.col("id") * 20).alias("subject"),
        (f.rand() > 0.66).cast("int").alias("y"),
    )


def test_knowledge_graph_sampler(
    original_knowledge_graph_nodes: ps.DataFrame,
    original_knowledge_graph_edges: ps.DataFrame,
    ground_truth_edges: ps.DataFrame,
):
    sampler = GroundTruthRandomSampler(
        knowledge_graph_nodes_sample_ratio=0.1,
        ground_truth_edges_sample_ratio=0.1,
        seed=42,
    )

    output = sampler.sample(original_knowledge_graph_nodes, original_knowledge_graph_edges, ground_truth_edges)

    assert output["nodes"].count() == pytest.approx(50, rel=0.2)
    assert output["edges"].count() == pytest.approx(60, rel=0.2)
