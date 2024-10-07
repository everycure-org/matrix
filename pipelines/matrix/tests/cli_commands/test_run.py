import pytest

from matrix.cli_commands.run import RunConfig, _extract_config, _filter_nodes_missing_tag, _get_feed_dict, _run
from unittest.mock import MagicMock, Mock, patch
from kedro.io import DataCatalog


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


def test_filter_nodes_no_without_tags():
    """Test when there are no tags to filter (without_tags is empty)."""
    without_tags = []
    pipeline = "test_pipeline"
    node_names = ["node1", "node2", "node3"]

    pipeline = MagicMock()
    pipeline.nodes = [MagicMock(name="node1"), MagicMock(name="node2"), MagicMock(name="node3")]

    result = _filter_nodes_missing_tag(without_tags, pipeline, node_names)

    assert result == node_names


def test_filter_nodes_all_without_tags():
    """Test when all nodes have the tag to be filtered."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    node_names = ["node1", "node2"]

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag1"}

    pipeline.nodes = [node1, node2]

    with pytest.raises(SystemExit):  # Expecting an exit since all nodes are filtered out
        _filter_nodes_missing_tag(without_tags, pipeline, node_names)


def test_filter_nodes_some_without_tags():
    """Test when only some nodes have the tag to be filtered."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
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

    result = _filter_nodes_missing_tag(without_tags, pipeline, node_names)

    assert result == ["node2"]


def test_filter_nodes_downstream_removal():
    """Test that downstream nodes are also removed."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
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

    result = _filter_nodes_missing_tag(without_tags, pipeline, node_names)

    # node1 is removed due to the tag and node2 is its downstream, so it should also be removed
    assert result == ["node3"]


def test_filter_nodes_empty_node_names():
    """Test when the node_names list is empty, should consider all nodes in the pipeline."""
    without_tags = ["tag1"]
    pipeline = MagicMock()
    node_names = []

    node1 = MagicMock()
    node1.name = "node1"
    node1.tags = {"tag1"}

    node2 = MagicMock()
    node2.name = "node2"
    node2.tags = {"tag2"}

    pipeline.nodes = [node1, node2]

    result = _filter_nodes_missing_tag(without_tags, pipeline, node_names)

    # node1 is removed due to the tag
    assert result == ["node2"]


def test_run_basic():
    config = RunConfig(
        pipeline_obj=None,
        pipeline_name="",
        env="test",
        runner="SequentialRunner",
        is_async=False,
        node_names=["node1"],
        to_nodes=[],
        from_nodes=[],
        from_inputs=[],
        to_outputs=[],
        load_versions=[],
        tags=[],
        without_tags=[],
        conf_source=None,
        params={},
        from_env=None,
    )
    # Create the mock for kedro_session and its method 'create'
    kedro_session = MagicMock()
    kedro_session.create.return_value.__enter__.return_value.run = MagicMock()

    _run(config, kedro_session)

    # Assert that the Kedro session was started
    kedro_session.create.assert_called_once_with(env="test", conf_source=None, extra_params={})

    # Assert that session.run() was called once
    kedro_session.create.return_value.__enter__.return_value.run.assert_called_once()


def test_run_with_fabricator_env_error():
    """Test that the _run function raises an error when running 'fabricator' in 'base' environment."""
    with pytest.raises(RuntimeError, match="might overwrite production data!"):
        _run(
            config=RunConfig(
                pipeline_obj=None,
                pipeline_name="fabricator",
                env="base",
                runner="SequentialRunner",
                is_async=False,
                node_names=["node1"],
                to_nodes=[],
                from_nodes=[],
                from_inputs=[],
                to_outputs=[],
                load_versions=[],
                tags=[],
                without_tags=[],
                conf_source=None,
                params={},
                from_env=None,
            ),
            kedro_session=MagicMock(),
        )


@pytest.fixture
def mock_config():
    return RunConfig(
        pipeline_obj=Mock(),
        pipeline_name="test_pipeline",
        env="test",
        runner="SequentialRunner",
        is_async=False,
        node_names=[],
        to_nodes=[],
        from_nodes=[],
        from_inputs=[],
        to_outputs=[],
        load_versions=[],
        tags=[],
        without_tags=[],
        conf_source="",
        params={},
        from_env="test_from_env",
    )


@pytest.fixture
def mock_session():
    session = Mock()
    session._conf_source = "test_conf_source"
    session._project_path = "/test/project/path"
    return session


@patch("matrix.cli_commands.run.settings")
def test_extract_config_with_from_env(mock_settings, mock_config, mock_session):
    # Setup mocks
    mock_config_loader = MagicMock()
    mock_config_loader.__getitem__.side_effect = lambda key: {
        "catalog": {"test_dataset": {"type": "MemoryDataSet"}},
        "credentials": {"test_cred": "secret"},
        "parameters": {"param1": "value1"},
    }[key]

    mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
    mock_settings.CONFIG_LOADER_ARGS = {}
    mock_settings.DATA_CATALOG_CLASS.from_config.return_value = Mock(spec=DataCatalog)

    # Call the function
    result = _extract_config(mock_config, mock_session)

    # Assertions
    assert result is not None
    assert isinstance(result, DataCatalog)
    mock_settings.CONFIG_LOADER_CLASS.assert_called_once_with(conf_source="test_conf_source", env="test_from_env")
    mock_settings.DATA_CATALOG_CLASS.from_config.assert_called_once_with(
        catalog={"test_dataset": {"type": "MemoryDataSet"}}, credentials={"test_cred": "secret"}
    )
    result.add_feed_dict.assert_called_once_with({"params:param1": "value1"}, replace=True)


def test_extract_config_without_from_env(mock_config, mock_session):
    mock_config.from_env = None
    result = _extract_config(mock_config, mock_session)
    assert result is None


@patch("matrix.cli_commands.run.settings")
def test_extract_config_with_empty_catalog(mock_settings, mock_config, mock_session):
    mock_config_loader = Mock()
    mock_config_loader.__getitem__.side_effect = lambda key: {"catalog": {}, "credentials": {}, "parameters": {}}[key]

    mock_settings.CONFIG_LOADER_CLASS.return_value = mock_config_loader
    mock_settings.CONFIG_LOADER_ARGS = {}
    mock_settings.DATA_CATALOG_CLASS.from_config.return_value = Mock(spec=DataCatalog)

    result = _extract_config(mock_config, mock_session)

    assert result is not None
    assert isinstance(result, DataCatalog)
    result.add_feed_dict.assert_called_once_with({}, replace=True)
