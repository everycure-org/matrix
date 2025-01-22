"""Script used in conjunction with the CICD system to flag the type of release."""

import os

import semver


def get_generate_notes_flag():
    latest_official_release = os.getenv("latest_official_release", "v0.0.0").lstrip("v")
    tag_version = semver.Version.parse(latest_official_release)

    release = os.getenv("release", "v0.0.0").lstrip("v")
    release_version = semver.Version.parse(release)

    if (
        tag_version.major == release_version.major
        and tag_version.minor == release_version.minor
        and tag_version.patch < release_version.patch
    ):
        # This is a patch bump
        generate_notes = str(False)
    else:
        generate_notes = str(True)

    print(f"{generate_notes=}")


if __name__ == "__main__":
    get_generate_notes_flag()
