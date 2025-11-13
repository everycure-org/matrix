import logging
import re
import subprocess
from typing import List

import semver

logger = logging.getLogger(__name__)


BRANCH_NAME_REGEX = r"^release/v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"


def get_git_sha() -> str:
    """Returns the git commit sha"""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return sha


def get_current_git_branch() -> str:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return branch


def get_current_git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def get_changed_git_files() -> list[str]:
    """Checks for uncommitted or untracked files. Empty list return means no such files were found"""
    return subprocess.check_output(["git", "status", "--porcelain"], text=True).strip().split("\n")


def has_legal_branch_name() -> bool:
    """Checks if branch conforms to the pattern of release/v{semver} or release/v{semver}-{suffix}"""
    branch = get_current_git_branch()
    match = re.match(BRANCH_NAME_REGEX, branch)
    return bool(match)


def has_unpushed_commits() -> bool:
    try:
        result = subprocess.run(
            ["git", "log", "@{upstream}.."], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
    except subprocess.CalledProcessError as e:
        if "no upstream configured for branch" in e.stderr:
            logger.exception(
                "You have not pushed your changes. The remote needs them so we can always checkout the commit from which a release was triggered."
            )
        raise
    local_commits = bool(result.stdout)
    return bool(local_commits)


def git_tag_exists(tag: str) -> bool:
    result = subprocess.check_output(f"git ls-remote --tags origin {tag}", shell=True, text=True)
    return tag in result


def get_tags() -> List[str]:
    result = subprocess.run(["git", "ls-remote", "--tags", "origin"], check=True, capture_output=True, text=True)
    return [
        line.split("\t")[1].replace("refs/tags/", "")
        for line in result.stdout.strip().split("\n")
        if not line.split("\t")[1].endswith("^{}")  # exclude dereferenced annotated tags
    ]


def get_latest_minor_release(releases_list: List[str]) -> str:
    original_to_mapped = correct_non_semver_compliant_release_names(releases_list)
    parsed_versions = [semver.Version.parse(v.lstrip("v")) for v in original_to_mapped]
    latest_major_minor = max(parsed_versions)
    # Find the earliest release in the latest major-minor series.
    latest_minor_release = min(
        [v for v in parsed_versions if v.major == latest_major_minor.major and v.minor == latest_major_minor.minor]
    )

    return original_to_mapped[f"v{latest_minor_release}"]


def correct_non_semver_compliant_release_names(releases_list: List[str]) -> dict[str, str]:
    """Map versions that aren't semver compliant to compliant ones."""
    mapper = {"v0.1": "v0.1.0", "v0.2": "v0.2.0"}
    original_to_mapped = {mapper.get(release, release): release for release in releases_list}
    return original_to_mapped


def abort_if_intermediate_release(release_version: str) -> None:
    release_version = semver.Version.parse(release_version.lstrip("v"))
    tags_list = get_tags()
    latest_minor_release = (get_latest_minor_release(tags_list)).lstrip("v").split(".")
    latest_major = int(latest_minor_release[0])
    latest_minor = int(latest_minor_release[1])
    if (
        release_version.major == latest_major and release_version.minor < latest_minor
    ) or release_version.major < latest_major:
        raise ValueError("Cannot release a minor/major version lower than the latest official release")
