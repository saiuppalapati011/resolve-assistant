import pytest
from unittest.mock import MagicMock, patch
from backend.agent.nodes.router import run
import pytest

class TestRouter:
    @pytest.mark.asyncio
    @patch("backend.agent.nodes.router.get_provider")
    async def test_router_action(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = {"content": "action"}
        mock_get_provider.return_value = mock_provider
        
        state = {"messages": [{"role": "user", "content": "Create a new timeline called Edit"}], "llm_provider": "anthropic", "llm_model": "test"}
        new_state = await run(state)
        
        assert new_state["intent"] == "action"



    @pytest.mark.asyncio
    @patch("backend.agent.nodes.router.get_provider")
    async def test_router_qa(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = {"content": "qa"}
        mock_get_provider.return_value = mock_provider
        
        state = {"messages": [{"role": "user", "content": "How do I use color wheels?"}], "llm_provider": "anthropic", "llm_model": "test"}
        new_state = await run(state)
        
        assert new_state["intent"] == "qa"

    @pytest.mark.asyncio
    @patch("backend.agent.nodes.router.get_provider")
    async def test_router_fallback(self, mock_get_provider):
        """On LLM failure the router should default to 'action', not 'qa'.
        'action' is the safer fallback: the planner will produce an out_of_scope
        response for genuinely ambiguous messages rather than silently answering
        from docs when the user might have wanted a live-state query.
        """
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = Exception("LLM failure")
        mock_get_provider.return_value = mock_provider

        state = {"messages": [{"role": "user", "content": "Hello"}], "llm_provider": "anthropic", "llm_model": "test"}
        new_state = await run(state)

        assert new_state["intent"] == "action"
