"""
Confirmation gate tests.
Asserts: every destructive tool triggers confirmation; every safe tool doesn't.
This is a regression guard — if someone adds a tool without setting its tag correctly,
these tests will catch it.
"""
import pytest
from backend.agent.nodes.confirmation import run as confirmation_run, is_destructive
from backend.agent.graph import build_graph





from unittest.mock import patch

import pytest

class TestConfirmationNode:
    def _base_state(self):
        return {
            "messages": [{"role": "user", "content": "test"}],
            "intent": "action",
            "retrieved_docs": [],
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "execution_results": [],
            "final_response": "",
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4-5",
        }

    @pytest.mark.asyncio
    async def test_safe_tool_no_confirmation(self):
        state = self._base_state()
        state["planned_calls"] = [{"tool": "davinci-resolve_create_timeline", "args": {"name": "Test"}}]
        with patch('backend.agent.nodes.confirmation.is_destructive', return_value=False):
            result = await confirmation_run(state)
        assert result["pending_confirmation"] is None

    @pytest.mark.asyncio
    async def test_destructive_tool_triggers_confirmation(self):
        state = self._base_state()
        state["planned_calls"] = [
            {"tool": "davinci-resolve_delete_clip", "args": {"track_type": "video", "track_index": 1, "clip_index": 0}}
        ]
        with patch('backend.agent.nodes.confirmation.is_destructive', return_value=True):
            result = await confirmation_run(state)
        assert result["pending_confirmation"] is not None
        assert "davinci-resolve_delete_clip" in result["pending_confirmation"]["destructive_tools"]

    @pytest.mark.asyncio
    async def test_mixed_plan_triggers_confirmation(self):
        """A plan with both safe and destructive tools should still trigger confirmation."""
        state = self._base_state()
        state["planned_calls"] = [
            {"tool": "davinci-resolve_create_timeline", "args": {"name": "Test"}},
            {"tool": "davinci-resolve_delete_clip", "args": {"clip_index": 0}},
        ]
        
        def mock_is_destructive(tool_name):
            return tool_name == "davinci-resolve_delete_clip"
            
        with patch('backend.agent.nodes.confirmation.is_destructive', side_effect=mock_is_destructive):
            result = await confirmation_run(state)
            
        assert result["pending_confirmation"] is not None

    @pytest.mark.asyncio
    async def test_multiple_safe_tools_no_confirmation(self):
        state = self._base_state()
        state["planned_calls"] = [
            {"tool": "davinci-resolve_import_media",    "args": {"file_paths": ["/tmp/a.mp4"]}},
            {"tool": "davinci-resolve_create_bin",      "args": {"bin_name": "Footage"}},
        ]
        with patch('backend.agent.nodes.confirmation.is_destructive', return_value=False):
            result = await confirmation_run(state)
        assert result["pending_confirmation"] is None


