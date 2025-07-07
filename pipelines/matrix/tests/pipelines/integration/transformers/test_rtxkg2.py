import pyspark.sql.functions as f
import pytest
from matrix.pipelines.integration.transformers.rtxkg2 import filter_semmed
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


@pytest.fixture
def edges_df(spark):
    """Create sample edges dataframe."""
    schema = StructType(
        [
            StructField("subject", StringType(), True),
            StructField("object", StringType(), True),
            StructField("aggregator_knowledge_source", ArrayType(StringType()), True),
            StructField("publications", ArrayType(StringType()), True),
        ]
    )

    data = [
        # Semmed edges
        ("curie1", "curie2", ["infores:semmeddb"], ["pub1", "pub2", "pub3"]),
        ("curie3", "curie4", ["infores:semmeddb"], ["pub4"]),
        # Non-semmed edge
        ("curie5", "curie6", ["other_source"], ["pub7", "pub8"]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def curie_to_pmids(spark):
    """Create sample curie to PMIDs mapping dataframe."""
    schema = StructType([StructField("curie", StringType(), True), StructField("pmids", StringType(), True)])

    data = [("curie1", "[1, 2, 3, 4, 5]"), ("curie2", "[2, 3, 4, 5, 6]"), ("curie3", "[1, 2]"), ("curie4", "[3, 4]")]

    return spark.createDataFrame(data, schema)


def test_filter_semmed_publication_threshold(edges_df, curie_to_pmids):
    """Test filtering based on publication threshold."""
    result = filter_semmed(
        edges_df=edges_df,
        curie_to_pmids=curie_to_pmids,
        publication_threshold=2,
        ngd_threshold=0.0,  # Set low to focus on publication threshold
        limit_pmids=5,
    )

    # Check that edges with fewer publications than threshold are filtered out
    semmed_edges = result.filter(
        (f.size(f.col("aggregator_knowledge_source")) == 1)
        & (f.col("aggregator_knowledge_source").getItem(0) == f.lit("infores:semmeddb"))
    )
    assert semmed_edges.filter(f.size("publications") < 2).count() == 0


def test_filter_semmed_preserves_non_semmed(edges_df, curie_to_pmids):
    """Test that non-semmed edges are preserved."""
    result = filter_semmed(
        edges_df=edges_df,
        curie_to_pmids=curie_to_pmids,
        publication_threshold=10,  # Set high to filter out all semmed edges
        ngd_threshold=1.0,
        limit_pmids=5,
    )

    # Check that non-semmed edges are preserved
    non_semmed = result.filter(
        (f.size(f.col("aggregator_knowledge_source")) == 1)
        & (f.col("aggregator_knowledge_source").getItem(0) == f.lit("infores:semmeddb"))
    )
    assert (
        non_semmed.count()
        == edges_df.filter(
            ~(f.size(f.col("aggregator_knowledge_source")) == 1)
            & (f.col("aggregator_knowledge_source").getItem(0) == f.lit("infores:semmeddb"))
        ).count()
    )
