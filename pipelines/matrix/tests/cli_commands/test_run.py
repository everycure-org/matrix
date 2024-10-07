import pytest

from matrix.cli_commands.run import _get_feed_dict


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
