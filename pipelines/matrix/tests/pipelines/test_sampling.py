import pyspark.sql as ps
import pyspark.sql.functions as f
import pytest
from matrix.pipelines.sampling.samplers import GroundTruthSampler

original_knowledge_graph_nodes_count = 500
original_knowledge_graph_edges_count = 1000
original_ground_truth_positive_pair_count = 50
original_ground_truth_negative_pair_count = 100


@pytest.fixture(scope="module")
def original_knowledge_graph_nodes(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_nodes_count).select("id")


@pytest.fixture(scope="module")
def original_knowledge_graph_edges(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_edges_count).select(
        f.col("id").alias("source"), (f.col("id") % original_knowledge_graph_nodes_count).alias("target")
    )


@pytest.fixture(scope="module")
def original_ground_truth_positive_pair(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_ground_truth_positive_pair_count).select(
        f.col("id").alias("source"), (f.col("id") * 10).alias("target")
    )


@pytest.fixture(scope="module")
def original_ground_truth_negative_pair(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_ground_truth_negative_pair_count).select(
        f.col("id").alias("source"), (f.col("id") * 5).alias("target")
    )


@pytest.fixture(scope="module")
def original_embeddings_nodes(spark: ps.SparkSession) -> ps.DataFrame:
    return spark.range(original_knowledge_graph_nodes_count).select("id", f.lit([0.1, 0.2, 0.3]).alias("embedding"))


def test_ground_truth_sampler(
    original_knowledge_graph_nodes: ps.DataFrame,
    original_knowledge_graph_edges: ps.DataFrame,
    original_ground_truth_positive_pair: ps.DataFrame,
    original_ground_truth_negative_pair: ps.DataFrame,
    original_embeddings_nodes: ps.DataFrame,
):
    sampler = GroundTruthSampler(
        ground_truth_positive_sample_ratio=0.1,
        ground_truth_negative_sample_ratio=0.1,
        knowledge_graph_nodes_sample_ratio=0.1,
        seed=42,
    )

    (
        sampled_knowledge_graph_nodes,
        sampled_ground_truth_positive_pair,
        sampled_ground_truth_negative_pair,
        sampled_embeddings_nodes,
    ) = sampler.sample(
        original_knowledge_graph_nodes,
        original_knowledge_graph_edges,
        original_ground_truth_positive_pair,
        original_ground_truth_negative_pair,
        original_embeddings_nodes,
    )

    assert sampled_knowledge_graph_nodes.count() == pytest.approx(50, rel=0.2)
    assert sampled_ground_truth_positive_pair.count() == pytest.approx(5, abs=3)
    assert sampled_ground_truth_negative_pair.count() == pytest.approx(10, abs=3)
    assert sampled_embeddings_nodes.count() == pytest.approx(50, rel=0.2)
