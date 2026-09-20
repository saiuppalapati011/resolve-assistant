from types import SimpleNamespace

import pytest

import backend.resolve.mcp_client as mcp_client


class _FakeSession:
    def __init__(self, text: str, is_error: bool = False):
        self.text = text
        self.is_error = is_error

    async def call_tool(self, name, arguments):
        assert name == "get_current_project"
        assert arguments == {}
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text)],
            isError=self.is_error,
        )


@pytest.mark.asyncio
async def test_probe_accepts_a_live_resolve_read(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_session",
        _FakeSession("Resolve test project"),
    )

    assert await mcp_client.check_resolve_connection() is True


@pytest.mark.asyncio
async def test_probe_rejects_resolve_bridge_error(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_session",
        _FakeSession(
            "DaVinci Resolve error: DaVinci Resolve is not running."
        ),
    )

    assert await mcp_client.check_resolve_connection() is False


@pytest.mark.asyncio
async def test_probe_rejects_mcp_error_result(monkeypatch):
    monkeypatch.setattr(
        mcp_client,
        "_session",
        _FakeSession("internal failure", is_error=True),
    )

    assert await mcp_client.check_resolve_connection() is False

