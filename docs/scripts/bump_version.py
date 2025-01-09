import os
import sys
import semver


def bump_version(type: str):
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    version = semver.Version.parse(latest_tag)

    if type == "minor":
        new_version = version.bump_minor()
    elif type == "patch":
        new_version = version.bump_patch()

    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"release_version=v{new_version}\n")


if __name__ == "__main__":
    # Extract the type argument
    arg = sys.argv[1]
    # Parse the type
    bump_type = arg.split("=", 1)[1]
    bump_version(bump_type)
