import pytest

from matrix.cli_commands.run import _filter_nodes_missing_tag, _get_feed_dict, _run
from unittest.mock import MagicMock, patch


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


@pytest.fixture
def kedro_session_mock():
    with patch("matrix.session.KedroSessionWithFromCatalog") as mock_session:
        yield mock_session


@pytest.fixture
def settings_mock():
    """Mock the settings object."""
    with patch("your_module.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def _filter_nodes_missing_tag_mock():
    """Mock the _filter_nodes_missing_tag function."""
    with patch("your_module._filter_nodes_missing_tag") as mock_filter_nodes:
        yield mock_filter_nodes


def test_run_basic(kedro_session_mock, pipelines_mock, _filter_nodes_missing_tag_mock):
    """Test running the basic _run function."""
    # Setup mock objects
    pipelines_mock["__default__"].nodes = []

    _run(
        pipeline=None,
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

    # Assert that the Kedro session was started
    kedro_session_mock.create.assert_called_once_with(env="test", conf_source=None, extra_params={})

    # Check that the runner was loaded
    load_obj.assert_called_once_with("SequentialRunner", "kedro.runner")

    # Ensure that the filtering function was called
    _filter_nodes_missing_tag_mock.assert_called_once_with(
        (), pipelines_mock["__default__"], kedro_session_mock.create(), ("node1",)
    )

    # Ensure that session.run was called with expected parameters
    kedro_session_mock.create.return_value.run.assert_called_once()


def test_run_with_fabricator_env_error():
    """Test that the _run function raises an error when running 'fabricator' in 'base' environment."""
    with pytest.raises(RuntimeError, match="might overwrite production data!"):
        _run(
            pipeline="fabricator",
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
        )


def test_run_with_from_env(
    kedro_session_mock, load_obj_mock, settings_mock, pipelines_mock, _filter_nodes_missing_tag_mock
):
    """Test _run when from_env is provided."""
    config_loader_mock = MagicMock()
    settings_mock.CONFIG_LOADER_CLASS.return_value = config_loader_mock

    _run(
        pipeline=None,
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
        from_env="custom_env",
    )

    # Ensure a second config loader was created for the 'from_env'
    config_loader_mock.assert_called_once_with(
        conf_source=kedro_session_mock.create.return_value._conf_source,
        env="custom_env",
        **settings_mock.CONFIG_LOADER_ARGS,
    )

    # Ensure from_catalog was loaded and passed to the session.run call
    kedro_session_mock.create.return_value.run.assert_called_once()
    assert "from_catalog" in kedro_session_mock.create.return_value.run.call_args.kwargs


def test_run_with_async_flag(kedro_session_mock, load_obj_mock, pipelines_mock):
    """Test the _run function with the async flag."""
    _run(
        pipeline=None,
        env="test",
        runner="SequentialRunner",
        is_async=True,
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

    # Ensure that the runner is executed with async flag
    load_obj_mock.assert_called_once_with("SequentialRunner", "kedro.runner")
    kedro_session_mock.create.return_value.run.assert_called_once()
    assert kedro_session_mock.create.return_value.run.call_args.kwargs["runner"].is_async is True


def test_run_filter_nodes(kedro_session_mock, load_obj_mock, _filter_nodes_missing_tag_mock, pipelines_mock):
    """Test that nodes are filtered correctly based on tags."""
    pipelines_mock["__default__"].nodes = []

    _run(
        pipeline=None,
        env="test",
        runner="SequentialRunner",
        is_async=False,
        node_names=["node1", "node2"],
        to_nodes=[],
        from_nodes=[],
        from_inputs=[],
        to_outputs=[],
        load_versions=[],
        tags=[],
        without_tags=["tag1"],
        conf_source=None,
        params={},
        from_env=None,
    )

    # Ensure that the node filtering function is called
    _filter_nodes_missing_tag_mock.assert_called_once_with(
        ("tag1",), pipelines_mock["__default__"], kedro_session_mock.create(), ("node1", "node2")
    )
