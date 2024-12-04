import subprocess


def get_git_sha() -> str:
    """
    Get the commit sha of the branch currently active when submitting a pipeline.
    Used as a label on the workflow that is being submitted.
    """
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return git_sha


def is_git_dirty() -> bool:
    """Returns true if uncommited or untracked changes are present in git."""
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return bool(status)
