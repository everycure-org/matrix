import os
from unittest.mock import patch

import pytest
from matrix.utils.hook_utilities import determine_hooks_to_execute


class TestDetermineHooksToExecute:
    @pytest.fixture
    def sample_hooks(self):
        """Sample hooks dictionary for testing."""
        return {
            "hook1": lambda: "hook1",
            "hook2": lambda: "hook2",
        }

    def test_given_no_disabled_hooks_when_determining_hooks_then_returns_all_hooks(self, sample_hooks):
        # Given
        with patch.dict(os.environ, {}, clear=True):
            # When
            result = determine_hooks_to_execute(sample_hooks)

            # Then
            assert len(result) == 2
            assert all(hook in result for hook in sample_hooks.values())

    def test_given_disabled_hook_when_determining_hooks_then_returns_only_enabled_hooks(self, sample_hooks):
        # Given
        with patch.dict(os.environ, {"KEDRO_HOOKS_DISABLE_HOOK1": "true"}, clear=True):
            # When
            result = determine_hooks_to_execute(sample_hooks)

            # Then
            assert len(result) == 1
            assert sample_hooks["hook2"] in result
            assert sample_hooks["hook1"] not in result
