import re
import subprocess

BRANCH_NAME_REGEX = r"^release/v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"


def get_git_sha() -> str:
    """Returns the git commit sha"""
    sha = subprocess.check_output(["git", "describe", "--no-match", "--always", "--abbrev=40"], text=True).strip()
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
        ["git", "log", "@{u}.."], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    local_commits = bool(result.stdout)
    return bool(local_commits)
