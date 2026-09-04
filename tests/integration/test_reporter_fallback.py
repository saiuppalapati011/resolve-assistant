import pytest
from unittest.mock import patch
from backend.agent.nodes.reporter import run as reporter_run
from backend.agent.state import AgentState

@pytest.mark.asyncio
async def test_reporter_deterministic_fallback():
    """
    Test that if execution_results contains a failure, and the LLM 
    generates a summary that omits the failure, the deterministic 
    fallback is triggered.
    """
    state = {
        "messages": [{"role": "user", "content": "do a multi-step action"}],
        "llm_provider": "anthropic",
        "llm_model": "test",
        "planned_calls": [{"tool": "step1"}, {"tool": "step2"}],
        "execution_results": [
            {"tool": "step1", "success": True, "message": "Done step 1", "data": {}},
            {"tool": "step2", "success": False, "message": "Failed step 2", "data": {}}
        ],
        "cancel_requested": None,
    }

    with patch('backend.agent.nodes.reporter.get_provider') as mock_get_provider:
        mock_provider = mock_get_provider.return_value
        # The LLM generates a summary that completely omits the failure
        mock_provider.generate.return_value = {"content": "I successfully completed all your actions!"}

        result_state = await reporter_run(state)

        # The final_response should be the deterministic fallback because the LLM omitted the failure keywords
        assert "I successfully completed all your actions!" not in result_state["final_response"]
        assert "I wasn't able to complete everything" in result_state["final_response"]
        assert "Failed step 2" in result_state["final_response"]
        assert result_state["technical_details"] == state["execution_results"]

@pytest.mark.asyncio
async def test_reporter_accepts_good_llm_summary():
    """
    Test that if execution_results contains a failure, and the LLM 
    generates a summary that correctly mentions the failure, it is accepted.
    """
    state = {
        "messages": [{"role": "user", "content": "do a multi-step action"}],
        "llm_provider": "anthropic",
        "llm_model": "test",
        "planned_calls": [{"tool": "step1"}],
        "execution_results": [
            {"tool": "step1", "success": False, "message": "Failed step 1", "data": {}}
        ],
        "cancel_requested": None,
    }

    with patch('backend.agent.nodes.reporter.get_provider') as mock_get_provider:
        mock_provider = mock_get_provider.return_value
        # The LLM mentions 'could not'
        mock_provider.generate.return_value = {"content": "I could not finish step 1 because it failed."}

        result_state = await reporter_run(state)

        # The final_response should be the LLM's summary
        assert "I could not finish step 1" in result_state["final_response"]
        assert "I wasn't able to complete everything" not in result_state["final_response"]
        assert result_state["technical_details"] == state["execution_results"]
