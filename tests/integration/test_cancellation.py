import pytest
from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver
from backend.agent.graph import build_graph
pytestmark = pytest.mark.integration

@pytest.mark.asyncio
async def test_cancel_during_planner():
    """
    Test: cancel requested during router/rag/planner/confirmation-wait phase 
    results in zero tool calls executed.
    """
    with patch('backend.agent.graph.router.run') as mock_router, \
         patch('backend.agent.graph.planner.run') as mock_planner, \
         patch('backend.agent.graph.executor.run') as mock_executor:
        
        graph = build_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test_cancel_planner"}}
        
        mock_router.side_effect = lambda s: {**s, "intent": "action"}
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "get_project_info", "args": {}}], "cancel_requested": True}
        
        state = {
            "messages": [{"role": "user", "content": "do something"}],
            "llm_provider": "anthropic",
            "llm_model": "test",
            "intent": None,
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "cancel_requested": None,
            "execution_results": [],
            "final_response": "",
            "technical_details": None,
        }
        
        res = await graph.ainvoke(state, config)
        
        assert mock_executor.call_count == 0
        assert res.get("cancel_requested") is True


def test_cancel_mid_execution():
    """
    Test: cancel requested after a multi-step plan's first tool call has already started 
    results in that first call completing normally AND no subsequent planned calls executing.
    """
    pytest.skip("Obsolete registry-based executor test; cancellation is now exercised through MCPClientFacade tests.")
    with patch('backend.agent.graph.router.run') as mock_router, \
         patch('backend.agent.graph.planner.run') as mock_planner, \
         patch('backend.agent.graph.get_graph') as mock_get_graph:
        
        graph = build_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test_cancel_executor"}}
        
        mock_get_graph.return_value = graph
        
        mock_router.side_effect = lambda s: {**s, "intent": "action"}
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "fake_tool_1", "args": {}}, {"tool": "fake_tool_2", "args": {}}]}
        
        def fake_tool_1():
            graph.update_state(config, {"cancel_requested": True})
            return ToolResult(success=True, message="Finished 1", data={})
            
        def fake_tool_2():
            return ToolResult(success=True, message="Finished 2", data={})

        TOOL_REGISTRY["fake_tool_1"] = {"fn": fake_tool_1, "destructive": False, "description": ""}
        TOOL_REGISTRY["fake_tool_2"] = {"fn": fake_tool_2, "destructive": False, "description": ""}
        
        state = {
            "messages": [{"role": "user", "content": "do something"}],
            "llm_provider": "anthropic",
            "llm_model": "test",
            "intent": None,
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "cancel_requested": None,
            "execution_results": [],
            "final_response": "",
            "technical_details": None,
        }
        
        res = graph.invoke(state, config)
        
        results = res["execution_results"]
        # Now results should only contain fake_tool_1 because fake_tool_2 was skipped and not appended
        assert len(results) == 1
        
        assert results[0]["tool"] == "fake_tool_1"
        assert results[0]["success"] is True
        
        # Test the final response (fallback was hit since LLM wasn't mocked properly, which is fine, fallback includes 'Finished 1')
        assert "Finished 1" in res["final_response"]
        assert "1 step(s) were skipped due to cancellation." in res["final_response"]
