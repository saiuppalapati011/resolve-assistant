"""
Unit tests for the planner node and its ordinal validation logic.
"""
from backend.agent.nodes.planner import validate_ordinals

def test_validate_ordinals_correct_match():
    user_msg = "delete the third clip"
    planned_calls = [{"tool": "delete_clip_from_timeline", "args": {"item_index": 3}}]
    
    # Should return None (no error), as the LLM correctly mapped "third" to 2
    assert validate_ordinals(user_msg, planned_calls) is None
    assert planned_calls[0]["args"]["item_index"] == 3

def test_validate_ordinals_autocorrect():
    user_msg = "delete the first clip on video track 1"
    # LLM mistakenly puts index 2 instead of 1 for "first"
    planned_calls = [{"tool": "delete_clip_from_timeline", "args": {"item_index": 2, "track_index": 1}}]
    
    result = validate_ordinals(user_msg, planned_calls)
    assert result is None
    
    assert planned_calls[0]["args"]["item_index"] == 1
    assert planned_calls[0]["args"]["track_index"] == 1

def test_validate_ordinals_multiple_ambiguous():
    user_msg = "move the first clip to the second track"
    # LLM proposes item_index 1 and track_index 2
    planned_calls = [{"tool": "move_clip", "args": {"item_index": 1, "track_index": 2}}]
    
    # The validator must distinguish the clip ordinal from the track ordinal;
    # this is a valid, unambiguous request.
    assert validate_ordinals(user_msg, planned_calls) is None

def test_validate_ordinals_no_ordinals():
    user_msg = "delete clip index 2"
    planned_calls = [{"tool": "delete_clip_from_timeline", "args": {"clip_index": 2}}]
    
    # Should do nothing, return None
    assert validate_ordinals(user_msg, planned_calls) is None
    assert planned_calls[0]["args"]["clip_index"] == 2

from unittest.mock import patch, AsyncMock
from backend.agent.nodes.planner import run as planner_run
from backend.agent.state import AgentState
import pytest

@pytest.mark.asyncio
async def test_planner_missing_capability_flat_decline():
    """
    Regression test: When the LLM decides there is no matching tool,
    the planner should immediately return a flat decline without
    attempting live discovery (which was removed).
    """
    state = {
        "messages": [{"role": "user", "content": "do something impossible"}],
        "llm_provider": "anthropic",
        "llm_model": "test",
    }
    
    with patch('backend.agent.nodes.planner.get_provider') as mock_get_provider, \
         patch('backend.agent.nodes.planner._get_retriever'), \
         patch('backend.agent.nodes.planner.get_mcp_client') as mock_mcp:
         
        mock_provider = mock_get_provider.return_value
        # LLM returns no_matching_tool
        mock_provider.generate.return_value = {
            "content": '{"planned_calls": [], "reason": "no_matching_tool", "error": "Cannot do that."}'
        }
        
        mcp_client_instance = AsyncMock()
        mock_mcp.return_value = mcp_client_instance
        mcp_client_instance.get_tools.return_value = []
        mcp_client_instance.execute_tool.return_value = []
        
        result_state = await planner_run(state)
        
        assert "planned_calls" in result_state
        assert len(result_state["planned_calls"]) == 0
        assert "I don't currently have a tool that can do that." in result_state["final_response"]
        assert "Cannot do that." in result_state["final_response"]
        # Ensure it didn't inject a tool_proposal
        assert "tool_proposal" not in result_state
