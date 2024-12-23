import os
import semver
import sys

latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
tag_version = semver.Version.parse(latest_tag)

release = os.getenv("release", "v0.0.0").lstrip("v")
release_version = semver.Version.parse(release)

if int(tag_version.major) == int(release_version.major) and int(tag_version.minor) < int(release_version.minor):
    bump_type = "minor"
elif (
    int(tag_version.major) == int(release_version.major)
    and int(tag_version.minor) == int(release_version.minor)
    and int(tag_version.patch) < int(release_version.patch)
):
    bump_type = "patch"
elif int(tag_version.major) < int(release_version.major):
    bump_type = "major"
else:
    sys.exit(1)

with open(os.getenv("GITHUB_ENV"), "a") as env_file:
    env_file.write(f"bump_type={bump_type}\n")
