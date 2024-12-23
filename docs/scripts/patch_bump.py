import os
import semver


def patch_bump():
    latest_tag = os.getenv("latest_tag", "v0.0.0").lstrip("v")
    version = semver.Version.parse(latest_tag)
    new_version = version.bump_patch()
    with open(os.getenv("GITHUB_ENV"), "a") as env_file:
        env_file.write(f"release_version=v{new_version}\n")


if __name__ == "__main__":
    patch_bump()
