import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from backend.agent.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

class MockCallToolResult:
    def __init__(self, is_error=False, text="success"):
        self.isError = is_error
        self.content = [MagicMock(text=text)]

def test_mcp_domain_activation_no_confirmation():
    """
    Test that activating a domain bypassing the confirmation gate and executes properly.
    """
    with patch('backend.agent.nodes.router.run') as mock_router, \
         patch('backend.agent.nodes.planner.run') as mock_planner, \
         patch('backend.agent.nodes.executor.get_mcp_client') as mock_mcp, \
         patch('backend.agent.nodes.confirmation.is_destructive') as mock_is_destructive, \
         patch('backend.agent.graph.get_graph') as mock_get_graph:
        
        mock_router.side_effect = lambda s: {**s, "intent": "action"}
        # Planner outputs an activation call
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "davinci-resolve_activate_domain", "args": {"domain": "project_management"}}]}
        
        # is_destructive returns False for activate_domain
        mock_is_destructive.return_value = False
        
        mock_graph_instance = MagicMock()
        mock_graph_instance.aget_state = AsyncMock(return_value=MagicMock(values={}))
        mock_get_graph.return_value = mock_graph_instance
        
        mcp_client_instance = AsyncMock()
        mock_mcp.return_value = mcp_client_instance
        
        # Mock get_tools
        tool_mock = MagicMock(inputSchema={"properties": {"domain": {"enum": ["project_management"]}}})
        tool_mock.name = "davinci-resolve_activate_domain"
        mcp_client_instance.get_tools.return_value = [tool_mock]
        
        # Mock execute_tool via ainvoke
        tool_mock.ainvoke = AsyncMock(return_value=MockCallToolResult(is_error=False, text="Domain activated"))
        
        graph = build_graph().compile(checkpointer=MemorySaver())
        
        # Using ainvoke since graph natively supports async
        state = {
            "messages": [{"role": "user", "content": "activate project management"}],
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
        
        result = asyncio.run(graph.ainvoke(state, {"configurable": {"thread_id": "test_activation"}}))
        
        # Verify it went to executor, no confirmation
        assert result.get("pending_confirmation") is None
        assert len(result.get("execution_results", [])) == 1
        assert result["execution_results"][0]["success"] is True
        
        # Verify it went to reporter
        assert "Domain activated" in result["final_response"] or "success" in result["final_response"] or "Here is what happened" in result["final_response"]

def test_mcp_domain_activation_cancellation():
    """
    Test that a cancellation flag interrupts the execution.
    """
    with patch('backend.agent.nodes.router.run') as mock_router, \
         patch('backend.agent.nodes.planner.run') as mock_planner, \
         patch('backend.agent.nodes.executor.get_mcp_client') as mock_mcp, \
         patch('backend.agent.nodes.confirmation.is_destructive') as mock_is_destructive, \
         patch('backend.agent.graph.get_graph') as mock_get_graph:
        
        mock_router.side_effect = lambda s: {**s, "intent": "action"}
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "davinci-resolve_activate_domain", "args": {"domain": "project_management"}}]}
        mock_is_destructive.return_value = False
        
        mcp_client_instance = AsyncMock()
        mock_mcp.return_value = mcp_client_instance
        
        tool_mock = MagicMock(name="davinci-resolve_activate_domain", inputSchema={"properties": {"domain": {"enum": ["project_management"]}}})
        mcp_client_instance.get_tools.return_value = [tool_mock]
        tool_mock.ainvoke = AsyncMock(return_value=MockCallToolResult(is_error=False, text="Domain activated"))
        
        graph = build_graph().compile(checkpointer=MemorySaver())
        
        state = {
            "messages": [{"role": "user", "content": "activate project management"}],
            "intent": "action",
            "retrieved_docs": [],
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "cancel_requested": True, # This flag will trigger the cancellation loop
            "execution_results": [],
            "final_response": "",
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4-5",
        }
        
        result = asyncio.run(graph.ainvoke(state, {"configurable": {"thread_id": "test_cancel"}}))
        
        # Since cancel_requested is true before the executor runs, the executor will skip tools
        # Wait, the graph router will route to END if cancel_requested is True at the beginning!
        # Let's verify the graph behavior for router:
        # router conditional edge: lambda s: "end" if s.get("cancel_requested") else s["intent"]
        # So it shouldn't even reach the planner.
        assert result.get("execution_results", []) == []
        assert result.get("planned_calls", []) == []
