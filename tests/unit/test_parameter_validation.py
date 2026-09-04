import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from backend.agent.nodes.executor import run

def dummy_state(calls):
    return {"planned_calls": calls}

@pytest.fixture
def mock_mcp_env():
    # Setup mock tools as they would be returned from mcp_client.get_tools()
    tool_enum = MagicMock(inputSchema={
        "properties": {
            "color": {
                "type": "string",
                "enum": ["Red", "Green", "Blue"]
            }
        }
    })
    tool_enum.name = "mock_tool_with_enum"
    
    tool_no_enum = MagicMock(inputSchema={
        "properties": {
            "name": {
                "type": "string"
            }
        }
    })
    tool_no_enum.name = "mock_tool_without_enum"

    class MockResult:
        isError = False
        content = [MagicMock(text="Success")]
        
    tool_enum.ainvoke = AsyncMock(return_value=MockResult())
    tool_no_enum.ainvoke = AsyncMock(return_value=MockResult())

    return [tool_enum, tool_no_enum]

@pytest.mark.asyncio
@patch("backend.agent.graph.get_graph")
@patch("backend.agent.nodes.executor.get_mcp_client")
async def test_exact_match(mock_get_mcp, mock_get_graph, mock_mcp_env):
    mock_graph_instance = AsyncMock()
    mock_graph_instance.aget_state.return_value = MagicMock(values={})
    mock_get_graph.return_value = mock_graph_instance
    
    mcp_instance = AsyncMock()
    mock_get_mcp.return_value = mcp_instance
    mcp_instance.get_tools.return_value = mock_mcp_env
    

    
    state = dummy_state([{"tool": "mock_tool_with_enum", "args": {"color": "Red"}}])
    config = {"configurable": {"thread_id": "test"}}
    
    result = await run(state, config)
    exec_results = result["execution_results"]
    
    assert len(exec_results) == 1
    assert exec_results[0]["success"] is True
    # Should pass exact case
    assert exec_results[0]["args"]["color"] == "Red"

@pytest.mark.asyncio
@patch("backend.agent.graph.get_graph")
@patch("backend.agent.nodes.executor.get_mcp_client")
async def test_normalization(mock_get_mcp, mock_get_graph, mock_mcp_env):
    mock_graph_instance = AsyncMock()
    mock_graph_instance.aget_state.return_value = MagicMock(values={})
    mock_get_graph.return_value = mock_graph_instance
    
    mcp_instance = AsyncMock()
    mock_get_mcp.return_value = mcp_instance
    mcp_instance.get_tools.return_value = mock_mcp_env
    

    
    # "blue" (lowercase) -> Should be normalized to "Blue"
    state = dummy_state([{"tool": "mock_tool_with_enum", "args": {"color": " blue  "}}])
    config = {"configurable": {"thread_id": "test"}}
    
    result = await run(state, config)
    exec_results = result["execution_results"]
    
    assert len(exec_results) == 1
    assert exec_results[0]["success"] is True
    # Should be normalized
    assert exec_results[0]["args"]["color"] == "Blue"
    # Should record normalization in data
    assert exec_results[0]["data"]["original_args"]["color"] == " blue  "
    assert exec_results[0]["data"]["normalized_args"]["color"] == "Blue"

@pytest.mark.asyncio
@patch("backend.agent.graph.get_graph")
@patch("backend.agent.nodes.executor.get_mcp_client")
async def test_rejection(mock_get_mcp, mock_get_graph, mock_mcp_env):
    mock_graph_instance = AsyncMock()
    mock_graph_instance.aget_state.return_value = MagicMock(values={})
    mock_get_graph.return_value = mock_graph_instance
    
    mcp_instance = AsyncMock()
    mock_get_mcp.return_value = mcp_instance
    mcp_instance.get_tools.return_value = mock_mcp_env
    

    
    # "Yellow" -> Not in ["Red", "Green", "Blue"]
    state = dummy_state([{"tool": "mock_tool_with_enum", "args": {"color": "Yellow"}}])
    config = {"configurable": {"thread_id": "test"}}
    
    result = await run(state, config)
    exec_results = result["execution_results"]
    
    assert len(exec_results) == 1
    assert exec_results[0]["success"] is False
    assert "Invalid value 'Yellow' for 'color'" in exec_results[0]["message"]
    assert "Expected one of: Red, Green, Blue" in exec_results[0]["message"]
    assert mock_mcp_env[0].ainvoke.call_count == 0

@pytest.mark.asyncio
@patch("backend.agent.graph.get_graph")
@patch("backend.agent.nodes.executor.get_mcp_client")
async def test_unaffected_no_enum(mock_get_mcp, mock_get_graph, mock_mcp_env):
    mock_graph_instance = AsyncMock()
    mock_graph_instance.aget_state.return_value = MagicMock(values={})
    mock_get_graph.return_value = mock_graph_instance
    
    mcp_instance = AsyncMock()
    mock_get_mcp.return_value = mcp_instance
    mcp_instance.get_tools.return_value = mock_mcp_env
    

    
    state = dummy_state([{"tool": "mock_tool_without_enum", "args": {"name": "Anything"}}])
    config = {"configurable": {"thread_id": "test"}}
    
    result = await run(state, config)
    exec_results = result["execution_results"]
    
    assert len(exec_results) == 1
    assert exec_results[0]["success"] is True
    assert exec_results[0]["args"]["name"] == "Anything"
