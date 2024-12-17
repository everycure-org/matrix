import subprocess


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
