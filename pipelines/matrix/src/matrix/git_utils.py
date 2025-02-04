import re
import subprocess
from typing import TYPE_CHECKING, List

import semver

BRANCH_NAME_REGEX = r"^release/v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"


def get_git_sha() -> str:
    """Returns the git commit sha"""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return sha


def get_current_git_branch() -> str:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return branch


def has_dirty_git() -> bool:
    """Checks for uncommitted or untracked files. Empty string return means no such files were found"""
    is_dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return bool(is_dirty)


def has_legal_branch_name() -> bool:
    """Checks if branch conforms to the pattern of release/v{semver} or release/v{semver}-{suffix}"""
    branch = get_current_git_branch()
    match = re.match(BRANCH_NAME_REGEX, branch)
    return bool(match)


def has_unpushed_commits() -> bool:
    result = subprocess.run(
        ["git", "log", "@{upstream}.."], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    local_commits = bool(result.stdout)
    return bool(local_commits)


def git_tag_exists(tag: str) -> bool:
    result = subprocess.check_output(f"git ls-remote --tags origin {tag}", shell=True, text=True)
    return tag in result


def get_releases() -> List[str]:
    return (
        (subprocess.check_output(["gh", "release", "list", "--json", "tagName", "--jq", ".[].tagName"]))
        .decode("utf-8")
        .strip("\n")
        .split("\n")
    )


def get_latest_minor_release() -> str:
    mapped_releases, original_to_mapped = get_releases_with_semver_mapping()
    parsed_versions = [semver.Version.parse(v.lstrip("v")) for v in mapped_releases]
    latest_major_minor = max(parsed_versions, key=lambda v: (v.major, v.minor))
    latest_minor_release = min(
        [v for v in parsed_versions if v.major == latest_major_minor.major and v.minor == latest_major_minor.minor]
    )

    return original_to_mapped[f"v{latest_minor_release}"]


def get_releases_with_semver_mapping() -> tuple[list[str], dict[str, str]]:
    """Fetch releases and map non-semver versions to semver-compliant ones."""
    releases_list = get_releases()
    mapper = {"v0.1": "v0.1.0", "v0.2": "v0.2.0"}
    original_to_mapped = {mapper.get(release, release): release for release in releases_list}
    mapped_releases = list(original_to_mapped.keys())
    return mapped_releases, original_to_mapped
