# pylint: disable=missing-function-docstring,unused-argument
"""Module for unit tests in local storage."""

import os
from pathlib import Path

import pytest
from review_list.datasets.storage import LocalStorageService


def test_exists_false(tmpdir):
    # Given an empty storage service
    storage = LocalStorageService(tmpdir)

    # When invoking the exists function
    exists = storage.exists(Path("path/to/file.txt"))

    # Then exits is false
    assert not exists


def test_exists_true(tmpdir):
    # Given a storage service with a file saved
    storage = LocalStorageService(tmpdir)

    full_path = Path(tmpdir) / "path/to/file.txt"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with full_path.open("w+", encoding="utf-8") as file:
        file.write("foo,bar")

    # When invoking the exists function
    exists = storage.exists(Path("path/to/file.txt"))

    # Then exits is true
    assert exists


def test_list(tmpdir):
    # Given a storage service with files saved
    storage = LocalStorageService(tmpdir)

    paths = ["path/to/file.txt", "path/to/other_file.txt"]
    for path in paths:
        full_path = Path(tmpdir) / path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with full_path.open("w+", encoding="utf-8") as file:
            file.write("foo,bar")

    # When invoking the list function
    result = storage.ls("path/to/*")
    expected_result = [Path(path) for path in paths]

    # Then correct list returned
    assert sorted(result) == sorted(expected_result)


def test_list_glob(tmpdir):
    # Given a storage service with files saved
    storage = LocalStorageService(tmpdir)

    paths = ["path/to/file.txt", "path/from/file.txt"]
    for path in paths:
        full_path = Path(tmpdir) / path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with full_path.open("w+", encoding="utf-8") as file:
            file.write("foo,bar")

    # When invoking the list function with a glob
    result = storage.ls("path/*/file.txt")
    expected_result = [Path(path) for path in paths]

    # Then correct list returned
    assert sorted(result) == sorted(expected_result)


def test_save(tmpdir):
    # Given a storage service
    storage = LocalStorageService(tmpdir)

    # When storing a file
    path = storage.save(Path("path/to/file.txt"), "foo,bar")

    # Then file exists and contains correct contents
    assert path.exists()
    assert path == Path(tmpdir) / "path/to/file.txt"
    assert path.open(encoding="utf=8").read() == "foo,bar"


def test_save_exists(tmpdir):
    # Given a storage service with a file saved
    storage = LocalStorageService(tmpdir)
    storage.save(Path("path/to/file.txt"), "foo,bar")

    with pytest.raises(FileExistsError):
        # When attempting to store an additional file at the same path
        storage.save(Path("path/to/file.txt"), "foo,bar", overwrite=False)

        # Then FileExistsError raised


def test_save_exists_overwrite(tmpdir):
    # Given a storage service with a file saved
    storage = LocalStorageService(tmpdir)
    storage.save(Path("path/to/file.txt"), "foo,bar")

    # When attempting to store an additional file at the same path
    path = storage.save(Path("path/to/file.txt"), "foobar", overwrite=True)

    # Then file exists and contains correct contents
    assert path.exists()
    assert path.open(encoding="utf=8").read() == "foobar"
