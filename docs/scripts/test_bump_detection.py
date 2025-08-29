import pytest

from scripts.bump_detection import get_generate_article_flag


@pytest.mark.parametrize(
    "latest_official_release, release, expected_result",
    [
        ("v1.0.0", "v1.1.0", True),  # Minor bump → Should generate notes
        ("v1.0.0", "v2.0.0", True),  # Major bump → Should generate notes
        ("v1.0.0", "v1.0.1", False),  # Patch bump → No notes
        ("v0.2.5", "v0.2.3", False),  # Intermediate patch → No notes
    ],
)
def test_get_generate_notes_flag(latest_official_release, release, expected_result):
    assert get_generate_article_flag(latest_official_release, release) == expected_result


@pytest.mark.parametrize(
    "latest_official_release, release",
    [
        ("v2.2.0", "v2.1.0"),  # Minor downgrade
        ("v3.0.0", "v2.2.5"),  # Major downgrade
    ],
)
def test_get_generate_notes_flag_invalid_release(latest_official_release, release):
    """Test that a ValueError is raised for invalid releases."""
    with pytest.raises(ValueError, match="Cannot release a major/minor version lower than the latest official release"):
        get_generate_article_flag(latest_official_release, release)
