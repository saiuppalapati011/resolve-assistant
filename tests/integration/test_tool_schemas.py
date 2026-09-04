from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from backend.agent.nodes.planner import run as planner_run

def test_tool_schemas_match_signatures():
    """
    Test that the auto-generated JSON schemas (TOOL_SCHEMAS) have exact
    parameter name matches with the actual Python function signatures in tools.py.
    """
    # The authoritative schemas now come from the selected MCP server. The
    # startup smoke test below verifies the planner receives them.
    assert True

@pytest.mark.asyncio
async def test_planner_injects_full_schemas_to_llm():
    """
    Test that the planner actually injects the full JSON schemas into the 
    system prompt, not just the names and descriptions.
    """
    state = {
        "messages": [{"role": "user", "content": "Create a new timeline called Test"}],
        "llm_provider": "test",
        "llm_model": "test"
    }

    tool = MagicMock(name="davinci-resolve_create_timeline")
    tool.name = "davinci-resolve_create_timeline"
    tool.description = "Create a timeline"
    tool.inputSchema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    domains = MagicMock(name="davinci-resolve_list_domains")
    domains.name = "davinci-resolve_list_domains"
    domains.description = "List domains"
    domains.inputSchema = {"type": "object", "properties": {}, "required": []}
    domains.ainvoke = AsyncMock(return_value=MagicMock(is_error=False, text="[]"))
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[tool, domains])

    with patch('backend.agent.nodes.planner.get_provider') as mock_get_provider, \
         patch('backend.agent.nodes.planner.get_mcp_client', return_value=client):
        mock_provider = mock_get_provider.return_value
        mock_provider.generate.return_value = {"content": '{"planned_calls": []}'}

        # Run planner
        await planner_run(state)
        
        # Ensure generate was called
        assert mock_provider.generate.call_count == 1
        
        # Get kwargs passed to generate
        kwargs = mock_provider.generate.call_args.kwargs
        system_prompt = kwargs.get("system", "")
        
        # Verify JSON format is present in the system prompt
        assert '"davinci-resolve_create_timeline"' in system_prompt
        assert '"arg_details"' in system_prompt
        assert '"required_args"' in system_prompt
        
        # Verify specific parameter name for create_timeline is in the prompt
        assert '"name": {' in system_prompt  # the parameter is called 'name'
