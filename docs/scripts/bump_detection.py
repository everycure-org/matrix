"""Script used in conjunction with the CICD system to flag the type of release."""

import sys

import semver


def get_generate_article_flag(latest_official_release, release):
    latest_official_release = latest_official_release.lstrip("v")
    tag_version = semver.Version.parse(latest_official_release)

    release = release.lstrip("v")

    release_version = semver.Version.parse(release)

    release_is_minor_bump = tag_version.major == release_version.major and tag_version.minor < release_version.minor

    release_is_major_bump = tag_version.major < release_version.major

    generate_article = release_is_minor_bump or release_is_major_bump

    if (
        tag_version.major == release_version.major and tag_version.minor > release_version.minor
    ) or tag_version.major > release_version.major:
        raise ValueError("Cannot release a major/minor version lower than the latest official release")

    print(f"generate_article={generate_article}")
    return generate_article


if __name__ == "__main__":
    get_generate_article_flag(latest_official_release=sys.argv[1], release=sys.argv[2])
