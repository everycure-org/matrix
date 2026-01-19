"""Tests for the schema functions."""

from matrix_pandera.schemas import (
    get_matrix_edge_schema,
    get_matrix_node_schema,
    get_unioned_edge_schema,
    get_unioned_node_schema,
)


def test_get_matrix_node_schema_returns_schema():
    schema = get_matrix_node_schema()
    assert schema is not None
    assert "id" in schema.columns
    assert "category" in schema.columns


def test_get_matrix_edge_schema_returns_schema():
    schema = get_matrix_edge_schema()
    assert schema is not None
    assert "subject" in schema.columns
    assert "predicate" in schema.columns
    assert "object" in schema.columns


def test_get_unioned_node_schema_returns_schema():
    schema = get_unioned_node_schema()
    assert schema is not None
    assert "id" in schema.columns


def test_get_unioned_edge_schema_returns_schema():
    schema = get_unioned_edge_schema()
    assert schema is not None
    assert "primary_knowledge_sources" in schema.columns
    assert "subject" in schema.columns
