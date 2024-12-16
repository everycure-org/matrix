import pytest
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from matrix.pipelines.integration.rtxkg2 import filter_semmed


@pytest.fixture
def edges_df(spark):
    """Create sample edges dataframe."""
    schema = StructType(
        [
            StructField("subject", StringType(), True),
            StructField("object", StringType(), True),
            StructField("primary_knowledge_source", StringType(), True),
            StructField("publications", ArrayType(StringType()), True),
        ]
    )

    data = [
        # Semmed edges
        ("curie1", "curie2", "infores:semmeddb", ["pub1", "pub2", "pub3"]),
        ("curie3", "curie4", "infores:semmeddb", ["pub4"]),
        # Non-semmed edge
        ("curie5", "curie6", "other_source", ["pub7", "pub8"]),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def curie_to_pmids(spark):
    """Create sample curie to PMIDs mapping dataframe."""
    schema = StructType([StructField("curie", StringType(), True), StructField("pmids", StringType(), True)])

    data = [("curie1", "[1, 2, 3, 4, 5]"), ("curie2", "[2, 3, 4, 5, 6]"), ("curie3", "[1, 2]"), ("curie4", "[3, 4]")]

    return spark.createDataFrame(data, schema)


def test_filter_semmed_primary_key_validation(edges_df, curie_to_pmids, spark):
    """Test primary key validation on curie_to_pmids."""
    # Add duplicate curie to test primary key validation
    duplicate_data = [("curie1", "[1, 2, 3]")]
    duplicate_df = spark.createDataFrame(duplicate_data, curie_to_pmids.schema)
    invalid_curie_to_pmids = curie_to_pmids.union(duplicate_df)

    with pytest.raises(TypeError, match=r"Primary key.*has duplicate values"):
        filter_semmed(
            edges_df=edges_df,
            curie_to_pmids=invalid_curie_to_pmids,
            publication_threshold=2,
            ngd_threshold=0.5,
            limit_pmids=5,
        )


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
    semmed_edges = result.filter(f.col("primary_knowledge_source") == f.lit("infores:semmeddb"))
    assert semmed_edges.filter(f.size("publications") < 2).count() == 0


def test_filter_semmed_ngd_threshold(edges_df, curie_to_pmids):
    """Test filtering based on NGD threshold."""
    # Count initial semmed edges
    initial_count = edges_df.filter(f.col("primary_knowledge_source") == f.lit("infores:semmeddb")).count()

    result = filter_semmed(
        edges_df=edges_df,
        curie_to_pmids=curie_to_pmids,
        publication_threshold=1,
        ngd_threshold=0.8,
        limit_pmids=5,
    )

    # Verify that edges were filtered
    final_count = result.filter(f.col("primary_knowledge_source") == f.lit("infores:semmeddb")).count()

    assert final_count < initial_count  # Some edges should be filtered out
    assert final_count > 0  # But not all edges should be removed


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
    non_semmed = result.filter(f.col("primary_knowledge_source") != f.lit("infores:semmeddb"))
    assert non_semmed.count() == edges_df.filter(f.col("primary_knowledge_source") != f.lit("infores:semmeddb")).count()


def test_filter_semmed_pmid_limit(edges_df, curie_to_pmids):
    """Test that PMIDs are limited correctly."""
    result = filter_semmed(
        edges_df=edges_df,
        curie_to_pmids=curie_to_pmids,
        publication_threshold=1,
        ngd_threshold=0.0,
        limit_pmids=3,  # Only take first 3 PMIDs
    )

    # Join back with curie_to_pmids to check PMID arrays
    enriched = result.join(curie_to_pmids.withColumnRenamed("curie", "subject"), on="subject", how="left")

    # Verify no PMID array has more than 3 elements
    assert enriched.filter(f.size("pmids") > 3).count() == 0
