import os
import sys

import semver


def bump_version(type: str) -> None:
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    version = semver.Version.parse(latest_tag)

    if type == "minor":
        new_version = version.bump_minor()
    elif type == "patch":
        new_version = version.bump_patch()
    else:
        raise NotImplementedError()

    print(f"release_version=v{new_version}")


if __name__ == "__main__":
    # Extract the type argument, which looks like "--type=**"
    arg = sys.argv[1]
    # Parse the type
    bump_type = arg.split("=", 1)[1]
    bump_version(bump_type)
