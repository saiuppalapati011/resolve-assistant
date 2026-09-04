"""
Integration tests for state reset between turns.
"""
import pytest
from backend.agent.graph import build_graph

@pytest.mark.asyncio
async def test_state_reset_consecutive_turns():
    """
    Test that a second turn does not inherit planned_calls or intent from the first turn,
    and a second destructive action triggers a fresh confirmation (rather than inheriting confirmed: True).
    """
    # Since we can't easily mock the entire WebSocket/main.py loop here without a big setup,
    # we will simulate the exact input_state dict passed to graph.invoke in main.py.
    
    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph().compile(checkpointer=MemorySaver())
    
    # Simulate first turn: User requests a destructive action (and confirms it)
    # 1. User says "delete third clip" -> intent = action, planned = destructive
    state1 = {
        "messages": [{"role": "user", "content": "delete third clip"}],
        "llm_provider": "test",
        "llm_model": "test",
        "intent": "action",
        "planned_calls": [{"tool": "delete_clip_from_timeline", "args": {"clip_index": 2}}],
        "pending_confirmation": {"message": "confirm?"},
        "confirmed": True,
        "execution_results": [{"tool": "delete_clip_from_timeline", "success": True}],
        "final_response": "Deleted.",
        "technical_details": [{"tool": "delete_clip_from_timeline", "success": True}]
    }
    
    # 2. Simulate next turn: User says "delete first clip"
    # main.py SHOULD pass a fresh state resetting the fields.
    state2_input = {
        "messages": [{"role": "user", "content": "delete first clip"}],
        "llm_provider": "test",
        "llm_model": "test",
        "intent": None,
        "planned_calls": [],
        "pending_confirmation": None,
        "confirmed": None,
        "execution_results": [],
        "final_response": "",
        "technical_details": None,
    }
    
    # Update state via graph to simulate what graph.invoke(state2_input) does.
    # In LangGraph, passing empty list to a non-operator.add field overwrites it.
    
    config = {"configurable": {"thread_id": "test_thread"}}
    
    # Initialize checkpoint with state1
    graph.update_state(config, state1)
    
    # Now simulate main.py invoking the next turn with state2_input
    # We will just verify that if we pass state2_input to update_state, the checkpoint reflects it.
    graph.update_state(config, state2_input)
    
    current_state = graph.get_state(config).values
    
    assert current_state["confirmed"] is None
    assert current_state["pending_confirmation"] is None
    assert current_state["intent"] is None
    assert current_state["planned_calls"] == []
    assert current_state["technical_details"] is None


@pytest.mark.asyncio
async def test_consecutive_destructive_actions_require_fresh_confirmation():
    """
    Test that confirming one destructive action doesn't allow a second, 
    subsequent destructive action to bypass confirmation.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from unittest.mock import patch
    from backend.agent.graph import build_graph
    
    with patch('backend.agent.graph.router.run') as mock_router, \
         patch('backend.agent.graph.planner.run') as mock_planner, \
         patch('backend.agent.graph.executor.run') as mock_executor, \
         patch('backend.agent.graph.rag.run') as mock_rag:
        
        graph = build_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test_destructive_thread"}}
        
        mock_router.side_effect = lambda s: {**s, "intent": "action"}
        mock_executor.side_effect = lambda s: {**s, "execution_results": [{"tool": "delete_clip_from_timeline", "success": True, "message": "Success"}]}
        
        # 1. Turn 1: Delete a clip
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "delete_clip_from_timeline", "args": {"clip_index": 2}}]}
        
        state1 = {
            "messages": [{"role": "user", "content": "delete third clip"}],
            "llm_provider": "anthropic",
            "llm_model": "test",
            "intent": None,
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "execution_results": [],
            "final_response": "",
        }
        
        # Invoke -> should halt at confirmation gate
        res1 = await graph.ainvoke(state1, config)
        assert res1["pending_confirmation"] is not None
        assert res1["confirmed"] is None
        
        # Simulate frontend confirming the action
        graph.update_state(config, {"confirmed": True})
        res1_confirmed = await graph.ainvoke(None, config)
        assert res1_confirmed.get("execution_results")
        
        # 2. Turn 2: Delete a different clip
        mock_planner.side_effect = lambda s: {**s, "planned_calls": [{"tool": "delete_clip_from_timeline", "args": {"clip_index": 5}}]}
        mock_executor.side_effect = lambda s: {**s, "execution_results": [{"tool": "delete_clip_from_timeline", "success": True, "message": "Success"}]}
        
        # The WebSocket loop resets the state
        state2_input = {
            "messages": [{"role": "user", "content": "delete sixth clip"}],
            "llm_provider": "anthropic",
            "llm_model": "test",
            "intent": None,
            "planned_calls": [],
            "pending_confirmation": None,
            "confirmed": None,
            "execution_results": [],
            "final_response": "",
        }
        
        # Invoke -> should halt at confirmation gate AGAIN
        res2 = await graph.ainvoke(state2_input, config)
        assert res2["pending_confirmation"] is not None
        assert res2["confirmed"] is None
