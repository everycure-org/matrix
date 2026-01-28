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
    return [
        line
        for line in subprocess.check_output(["git", "status", "--porcelain"], text=True).strip().split("\n")
        if line.strip() != ""
    ]


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
    raw_tags = result.stdout.strip().split("\n")
    tags = [
        line.split("\t")[1].replace("refs/tags/", "") for line in raw_tags if not line.split("\t")[1].endswith("^{}")
    ]
    matrix_tags = [tag for tag in tags if tag.endswith("-matrix")]
    return matrix_tags


def get_latest_minor_release(tags_list: List[str]) -> str:
    parsed_versions = [semver.Version.parse(parse_release_version_from_matrix_tag(v)) for v in tags_list]
    latest_major_minor = max(parsed_versions)
    # Find the earliest release in the latest major-minor series.
    latest_minor_release = min(
        [v for v in parsed_versions if v.major == latest_major_minor.major and v.minor == latest_major_minor.minor]
    )

    return f"{latest_minor_release}"


def abort_if_intermediate_release(release_version: str) -> None:
    release_version = semver.Version.parse(parse_release_version_from_matrix_tag(release_version))
    tags_list = get_tags()
    release_list = [parse_release_version_from_matrix_tag(tag) for tag in tags_list]
    latest_minor_release = get_latest_minor_release(release_list).split(".")
    latest_major = int(latest_minor_release[0])
    latest_minor = int(latest_minor_release[1])
    if (
        release_version.major == latest_major and release_version.minor < latest_minor
    ) or release_version.major < latest_major:
        raise ValueError("Cannot release a minor/major version lower than the latest official release")


def parse_release_version_from_matrix_tag(release_version: str) -> str:
    return release_version.lstrip("v").rstrip("-matrix")
