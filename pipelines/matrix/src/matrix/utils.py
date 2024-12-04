import subprocess


def get_git_sha() -> str:
    """Returns the git commit sha suffixed with 'dirty' if uncommited or untracked files are present"""
    sha = subprocess.check_output(
        ["git", "describe", "--no-match", "--always", "--abbrev=40", "--dirty", "--broken"], text=True
    ).strip()
    return sha
