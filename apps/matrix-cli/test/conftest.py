from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_git_root(monkeypatch):
    """Mock the git root directory."""

    def mock_get_git_root():
        return Path("/fake/git/root")

    monkeypatch.setattr("matrix_cli.modules.code.get_git_root", mock_get_git_root)
    return "/fake/git/root"


@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run with configurable return values."""
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


@pytest.fixture
def mock_console(monkeypatch):
    """Mock rich console for capturing output."""
    mock_console = MagicMock()
    monkeypatch.setattr("matrix_cli.modules.code.console", mock_console)
    return mock_console


@pytest.fixture
def mock_rprint(monkeypatch):
    """Mock rich print function."""
    mock_print = MagicMock()
    monkeypatch.setattr("matrix_cli.modules.code.rprint", mock_print)
    return mock_print
