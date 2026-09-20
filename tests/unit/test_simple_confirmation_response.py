import pytest

import backend.main as main


@pytest.mark.asyncio
async def test_confirmed_turn_does_not_return_stale_confirmation(monkeypatch):
    async def fake_get_graph():
        return object()

    async def fake_cleanup(graph, config, result_state):
        return None

    async def fake_delete(thread_id):
        return None

    monkeypatch.setattr(main, "get_graph", fake_get_graph)
    monkeypatch.setattr(main, "_end_of_turn_cleanup", fake_cleanup)
    monkeypatch.setattr(main, "_delete_thread_state", fake_delete)

    payload = await main._simple_payload(
        "test-session",
        {
            "pending_confirmation": {"message": "Please confirm again"},
            "confirmed": True,
            "final_response": "Created timeline 'Test Main Time'.",
            "technical_details": [],
        },
    )

    assert payload["content"] == "Created timeline 'Test Main Time'."
    assert payload["needs_confirmation"] is None


@pytest.mark.asyncio
async def test_unconfirmed_turn_still_returns_confirmation(monkeypatch):
    payload = await main._simple_payload(
        "test-session",
        {
            "pending_confirmation": {"message": "Please confirm this action."},
            "confirmed": None,
            "final_response": "",
            "technical_details": [],
        },
    )

    assert payload["content"] == "Please confirm this action."
    assert payload["needs_confirmation"]["message"] == "Please confirm this action."
