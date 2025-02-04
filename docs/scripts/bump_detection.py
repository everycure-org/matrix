"""Script used in conjunction with the CICD system to flag the type of release."""

import os
import sys

import semver


def get_generate_notes_flag():
    latest_official_release = os.getenv("latest_official_release", "v0.0.0").lstrip("v")
    tag_version = semver.Version.parse(latest_official_release)

    release = os.getenv("release", "v0.0.0").lstrip("v")

    release_version = semver.Version.parse(release)

    release_is_minor_bump = tag_version.major == release_version.major and tag_version.minor < release_version.minor

    release_is_major_bump = tag_version.major < release_version.major

    generate_notes = release_is_minor_bump or release_is_major_bump

    if (
        tag_version.major == release_version.major and tag_version.minor > release_version.minor
    ) or tag_version.major > release_version.major:
        raise ValueError("Cannot release a major/minor version lower than the latest official release")

    print(f"generate_notes={generate_notes}")
    return generate_notes


if __name__ == "__main__":
    get_generate_notes_flag()
