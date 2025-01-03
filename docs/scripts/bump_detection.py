import os
import semver
import sys


def bump_type():
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    tag_version = semver.Version.parse(latest_tag)

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
        sys.exit(1)

    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"bump_type={bump_type}\n")


if __name__ == "__main__":
    bump_type()
