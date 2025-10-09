from matrix.pipelines.document_kg.nodes import (
    extract_pks_from_unified_edges,
    integrate_all_metadata,
    merge_all_pks_metadata,
)


def test_merge_multiple_sources():
    """Test merging metadata from multiple sources with overlapping PKS."""
    source1 = {"pks1": {"infores": {"id": "infores:pks1", "name": "Source 1"}}}
    source2 = {"pks1": {"reusabledata": {"license": "MIT"}}}
    source3 = {"pks2": {"infores": {"id": "infores:pks2", "name": "Source 2"}}}

    result = merge_all_pks_metadata(source1, source2, source3)

    assert len(result) == 2
    assert "infores" in result["pks1"]
    assert "reusabledata" in result["pks1"]
    assert result["pks1"]["infores"]["name"] == "Source 1"
    assert result["pks1"]["reusabledata"]["license"] == "MIT"


def test_merge_overlapping_sources():
    """Test that later sources update earlier sources."""
    source1 = {"pks1": {"infores": {"id": "infores:pks1", "name": "Original Name"}}}
    source2 = {"pks1": {"infores": {"name": "Updated Name"}}}

    result = merge_all_pks_metadata(source1, source2)

    assert result["pks1"]["infores"]["name"] == "Updated Name"
    assert result["pks1"]["infores"]["id"] == "infores:pks1"


def test_extract_from_all_columns(spark):
    """Test extraction from all PKS columns and deduplication."""
    data = [
        ("subj1", "pred1", "obj1", "infores:source1", None, None),
        ("subj2", "pred2", "obj2", "infores:source1", None, None),
        ("subj3", "pred3", "obj3", None, ["infores:source1", "infores:source2"], None),
        ("subj4", "pred4", "obj4", None, None, ["infores:agg1"]),
    ]

    df = spark.createDataFrame(
        data,
        [
            "subject",
            "predicate",
            "object",
            "primary_knowledge_source",
            "primary_knowledge_sources",
            "aggregator_knowledge_source",
        ],
    )

    result = extract_pks_from_unified_edges(df)

    assert "source1" in result
    assert "source2" in result
    assert "agg1" in result
    assert result.count("source1") == 1


def test_filters_to_relevant_sources(spark):
    """Test that only PKS found in unified edges are included."""
    all_metadata = {
        "source1": {"infores": {"id": "infores:source1", "name": "Source 1"}},
        "source2": {"infores": {"id": "infores:source2", "name": "Source 2"}},
        "source3": {"infores": {"id": "infores:source3", "name": "Source 3"}},
    }

    data = [
        ("subj1", "pred1", "obj1", "infores:source1", None, None),
        ("subj2", "pred2", "obj2", "infores:source2", None, None),
    ]

    unified_edges = spark.createDataFrame(
        data,
        [
            "subject",
            "predicate",
            "object",
            "primary_knowledge_source",
            "primary_knowledge_sources",
            "aggregator_knowledge_source",
        ],
    )

    result = integrate_all_metadata(all_metadata, unified_edges)

    assert len(result) == 2
    assert "source1" in result
    assert "source2" in result
    assert "source3" not in result


def test_creates_default_for_missing_metadata(spark):
    """Test that default entries are created for sources without metadata."""
    all_metadata = {"source1": {"infores": {"id": "infores:source1", "name": "Source 1"}}}

    data = [
        ("subj1", "pred1", "obj1", "infores:source1", None, None),
        ("subj2", "pred2", "obj2", "infores:source2", None, None),
    ]

    unified_edges = spark.createDataFrame(
        data,
        [
            "subject",
            "predicate",
            "object",
            "primary_knowledge_source",
            "primary_knowledge_sources",
            "aggregator_knowledge_source",
        ],
    )

    result = integrate_all_metadata(all_metadata, unified_edges)

    assert len(result) == 2
    assert result["source1"]["infores"]["name"] == "Source 1"
    assert "no metadata available" in result["source2"]["infores"]["name"].lower()
    assert result["source2"]["infores"]["status"] == "unknown"
