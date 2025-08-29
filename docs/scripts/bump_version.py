import argparse
import subprocess

import semver


def branch_exists(branch_name: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin", branch_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.returncode == 0


def bump_version(bump_type: str, latest_tag: str) -> None:
    latest_tag = latest_tag.lstrip("v")
    version = semver.Version.parse(latest_tag)
    while True:
        if bump_type == "minor":
            version = version.bump_minor()
        elif bump_type == "patch":
            version = version.bump_patch()
        else:
            raise NotImplementedError()

        new_branch = f"release/v{version}"
        if not branch_exists(new_branch):
            break

    print(f"release_version=v{version}")
    return f"v{version}"


if __name__ == "__main__":
    # Extract the type argument, which looks like "--type=**, --tag==**"
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--tag")

    args = parser.parse_args()

    bump_version(args.type, args.tag)
