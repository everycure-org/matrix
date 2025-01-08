import os
import semver


def bump_version():
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    bump_type = os.getenv("bump_type", "minor")
    version = semver.Version.parse(latest_tag)

    if bump_type == "minor":
        new_version = version.bump_minor()
    elif bump_type == "patch":
        new_version = version.bump_patch()

    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"release_version=v{new_version}\n")


if __name__ == "__main__":
    bump_version()
