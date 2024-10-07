import pytest

from matrix.cli_commands.run import _filter_nodes_missing_tag, _get_feed_dict


def test_get_feed_dict_simple():
    params = {"a": 1, "b": 2}
    expected_output = {
        "parameters": {"a": 1, "b": 2},
        "params:a": 1,
        "params:b": 2,
    }

    assert _get_feed_dict(params) == expected_output


def test_get_feed_dict_nested():
    params = {"a": {"b": 1, "c": 2}, "d": 3}
    expected_output = {
        "parameters": {"a": {"b": 1, "c": 2}, "d": 3},
        "params:a": {"b": 1, "c": 2},
        "params:a.b": 1,
        "params:a.c": 2,
        "params:d": 3,
    }

    assert _get_feed_dict(params) == expected_output


def test_get_feed_dict_empty():
    params = {}
    expected_output = {"parameters": {}}

    assert _get_feed_dict(params) == expected_output


def test_get_feed_dict_deeply_nested():
    params = {"a": {"b": {"c": {"d": 4}}}, "e": 5}
    expected_output = {
        "parameters": {"a": {"b": {"c": {"d": 4}}}, "e": 5},
        "params:a": {"b": {"c": {"d": 4}}},
        "params:a.b": {"c": {"d": 4}},
        "params:a.b.c": {"d": 4},
        "params:a.b.c.d": 4,
        "params:e": 5,
    }

    assert _get_feed_dict(params) == expected_output


# TODO(pascal.bro): Agree on validation
@pytest.mark.skip(reason="No validation implemented - agree if we want to enforce it.")
def test_get_feed_dict_non_string_keys():
    params = {1: "value", "key": 2}
    with pytest.raises(TypeError):
        _get_feed_dict(params)


def test_get_feed_dict_complex_values():
    params = {"a": [1, 2, 3], "b": {"c": [4, 5, 6]}}
    expected_output = {
        "parameters": {"a": [1, 2, 3], "b": {"c": [4, 5, 6]}},
        "params:a": [1, 2, 3],
        "params:b": {"c": [4, 5, 6]},
        "params:b.c": [4, 5, 6],
    }

    assert _get_feed_dict(params) == expected_output


import pytest
from unittest.mock import MagicMock


def test_filter_nodes_no_without_tags():
    """Test when there are no tags to filter (without_tags is empty)."""
    without_tags = []
    pipeline = "test_pipeline"
    session = MagicMock()
    node_names = ["node1", "node2", "node3"]

    pipeline = MagicMock()
    pipeline.nodes = [MagicMock(name="node1"), MagicMock(name="node2"), MagicMock(name="node3")]

    result = _filter_nodes_missing_tag(without_tags, pipeline, session, node_names)

    assert result == node_names


def test_filter_nodes_all_without_tags():
    """Test when all nodes have the tag to be filtered."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    session = MagicMock()
    node_names = ["node1", "node2"]

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag1"}

    pipeline.nodes = [node1, node2]

    with pytest.raises(SystemExit):  # Expecting an exit since all nodes are filtered out
        _filter_nodes_missing_tag(without_tags, pipeline, session, node_names)


def test_filter_nodes_some_without_tags():
    """Test when only some nodes have the tag to be filtered."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    session = MagicMock()
    node_names = ["node1", "node2", "node3"]

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag2"}

    node3 = MagicMock()
    node3.name = "node3"
    node3.tags = {"tag1", "tag2"}

    pipeline.nodes = [node1, node2, node3]

    result = _filter_nodes_missing_tag(without_tags, pipeline, session, node_names)

    assert result == ["node2"]


def test_filter_nodes_downstream_removal():
    """Test that downstream nodes are also removed."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    session = MagicMock()
    node_names = ["node1", "node2", "node3"]

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag2"}

    node3 = MagicMock()
    node3.name = "node3"
    node3.tags = {"tag2"}

    pipeline.nodes = [node1, node2, node3]
    pipeline.from_nodes.return_value = MagicMock()
    pipeline.from_nodes.return_value.nodes = [node2]

    result = _filter_nodes_missing_tag(without_tags, pipeline, session, node_names)

    # node1 is removed due to the tag and node2 is its downstream, so it should also be removed
    assert result == ["node3"]


def test_filter_nodes_empty_node_names():
    """Test when the node_names list is empty, should consider all nodes in the pipeline."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    session = MagicMock()  # Mock the Kedro session
    node_names = []

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag2"}

    pipeline.nodes = [node1, node2]

    result = _filter_nodes_missing_tag(without_tags, pipeline, session, node_names)

    # node1 is removed due to the tag
    assert result == ["node2"]
