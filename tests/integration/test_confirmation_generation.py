import pytest
from unittest.mock import patch
from backend.agent.nodes.confirmation import run as confirmation_run
from backend.agent.state import AgentState

@pytest.mark.asyncio
async def test_confirmation_generation():
    """
    Test that confirmation node uses LLM and sets up the correct payload structure.
    """

    state = {
        "messages": [{"role": "user", "content": "delete it"}],
        "llm_provider": "anthropic",
        "llm_model": "test",
        "planned_calls": [{"tool": "mock_delete_tool", "args": {"target": "clip1"}}],
        "cancel_requested": None,
    }

    with patch('backend.agent.nodes.confirmation.get_provider') as mock_get_provider, \
         patch('backend.agent.nodes.confirmation.is_destructive', return_value=True):
        mock_provider = mock_get_provider.return_value
        mock_provider.generate.return_value = {"content": "I'm about to delete clip1. Want me to go ahead?"}

        result_state = await confirmation_run(state)
        
        pending = result_state.get("pending_confirmation")
        assert pending is not None
        assert pending["message"] == "I'm about to delete clip1. Want me to go ahead?"
        assert pending["details"] == state["planned_calls"]
        assert pending["destructive_tools"] == ["mock_delete_tool"]
