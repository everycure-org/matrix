import os
import subprocess
import sys

import semver


def branch_exists(branch_name: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "exit-code", "orgin", branch_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.returncode == 0


def bump_version(bump_type: str) -> None:
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    version = semver.Version.parse(latest_tag)
    while True:
        if bump_type == "minor":
            new_version = version.bump_minor()
        elif bump_type == "patch":
            new_version = version.bump_patch()
        else:
            raise NotImplementedError()

        new_branch = f"release/v{new_version}"
        if not branch_exists(new_branch):
            break

        # Increment version again if branch exists
        version = semver.Version.parse(str(new_version))

    print(f"release_version=v{new_version}")


if __name__ == "__main__":
    # Extract the type argument, which looks like "--type=**"
    arg = sys.argv[1]
    # Parse the type
    bump_type = arg.split("=", 1)[1]
    bump_version(bump_type)
