"""
Integration tests — require a live DaVinci Resolve session.
Run with: pytest tests/integration/ -m integration -v

Resolve preconditions:
  - Resolve is open with external scripting enabled (Preferences > General > Local)
  - A project named "AI Test Project" is open
  - A timeline named "Main Timeline" exists
"""
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skip(reason="Legacy direct-Resolve tests; use the MCP live harness"),
]


@pytest.fixture(scope="module")
def resolve():
    """Connect to a live Resolve session."""
    from backend.resolve.connection import conn
    try:
        conn.connect(auto_launch=False)
    except ConnectionError as e:
        pytest.skip(f"Resolve not available: {e}")
    return conn


class TestLiveTimeline:
    def test_create_timeline(self, resolve):
        from backend.resolve.tools import create_timeline
        result = create_timeline("AI Test Timeline")
        assert result.success, result.message

    def test_add_marker(self, resolve):
        from backend.resolve.tools import add_marker
        result = add_marker(24, "Blue", "Integration test marker")
        assert result.success, result.message


class TestLiveMediaPool:
    def test_create_bin(self, resolve):
        from backend.resolve.tools import create_bin
        result = create_bin("AI Test Bin")
        assert result.success, result.message


class TestLiveRender:
    def test_get_render_status(self, resolve):
        from backend.resolve.tools import get_render_status
        result = get_render_status()
        assert result.success, result.message  # read-only, always succeeds
