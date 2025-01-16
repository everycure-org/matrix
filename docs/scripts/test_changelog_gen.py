import textwrap

import changelog_gen as mut
from unittest.mock import patch
from pathlib import Path
import pytest


@pytest.fixture()
def release_path(tmp_path: Path) -> Path:
    with patch("changelog_gen.locate_releases_path", return_value=tmp_path):
        yield tmp_path


def test_mocked_function():
    """Assert expectations on where the release info files exist aren't changed over time.

    If it was to change, thus making this test fail, some paths, e.g. those in
    docs/src/releases/release_history.md, may need to be updated.
    Other tests will be fine, since they work in a tmp dir."""

    assert mut.locate_releases_path() == Path(__file__).parents[1].resolve() / "src" / "releases" / "changelog_files"


def test_generate_changelog_aggregate_on_empty_dir(release_path):
    """When there are no release info files, a ValueError is raised."""

    with pytest.raises(ValueError, match="No release info found"):
        mut.main()


def test_generate_changelog_aggregate_from_files(release_path: Path):
    """Release info files containing properly formatted json are merged."""

    versions = ("v0.1.12", "v0.1.2")  # Note, the latter sorts semantically before the first
    filenames = tuple(f"{v}_info.json" for v in versions)
    contents = (
        f'{{"Release Name": "{versions[0]}",\n   "foo": "bar"}}',  # still proper json
        f'{{"Release Name": "{versions[1]}", "baz": "qux"}}',
    )
    for index in (0, 1):
        (release_path / filenames[index]).write_text(contents[index])

    mut.main()

    # Original files have not been altered
    for index in (0, 1):
        assert (release_path / filenames[index]).read_text() == contents[index]

    # Aggregate has been created, with releases sorted by version number.
    expected_contents = f"""\
    - Release Name: {versions[1]}
      baz: qux
    - Release Name: {versions[0]}
      foo: bar
    """
    assert (release_path / "releases_aggregated.yaml").read_text() == textwrap.dedent(expected_contents)
