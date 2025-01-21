import pyspark.sql as ps
import pyspark.sql.functions as f
import pytest
from matrix.pipelines.create_sample.samplers import KnowledgeGraphSampler

original_knowledge_graph_nodes_count = 500
original_knowledge_graph_edges_count = 1000


@pytest.fixture(scope="module")
def original_knowledge_graph_nodes(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_nodes_count).select("id")


@pytest.fixture(scope="module")
def original_knowledge_graph_edges(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_edges_count).select(
        f.col("id").alias("object"), (f.col("id") % original_knowledge_graph_nodes_count).alias("subject")
    )


def test_knowledge_graph_sampler(
    original_knowledge_graph_nodes: ps.DataFrame,
    original_knowledge_graph_edges: ps.DataFrame,
):
    sampler = KnowledgeGraphSampler(
        knowledge_graph_nodes_sample_ratio=0.1,
        seed=42,
    )

    (sampled_knowledge_graph_nodes, sampled_knowledge_graph_edges) = sampler.sample(
        original_knowledge_graph_nodes,
        original_knowledge_graph_edges,
    )

    assert sampled_knowledge_graph_nodes.count() == pytest.approx(50, rel=0.2)
    assert sampled_knowledge_graph_edges.count() == pytest.approx(60, rel=0.2)
