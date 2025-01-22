"""Script used in conjunction with the CICD system to flag the type of release."""

import os

import semver


def bump_type():
    latest_official_release = os.getenv("latest_official_release", "v0.0.0").lstrip("v")
    tag_version = semver.Version.parse(latest_official_release)

    release = os.getenv("release", "v0.0.0").lstrip("v")
    release_version = semver.Version.parse(release)

    if tag_version.major == release_version.major and tag_version.minor < release_version.minor:
        bump_type = "minor"
    elif (
        tag_version.major == release_version.major
        and tag_version.minor == release_version.minor
        and tag_version.patch < release_version.patch
    ):
        bump_type = "patch"
    elif tag_version.major < release_version.major:
        bump_type = "major"
    else:
        bump_type = "intermediate"

    if bump_type != "patch":
        generate_notes = True
    else:
        generate_notes = False

    # print(f"{bump_type=}")
    print(f"{generate_notes=}")


if __name__ == "__main__":
    bump_type()
