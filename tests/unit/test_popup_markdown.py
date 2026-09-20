import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "resolve_scripts" / "ResolveAssistant.py"
SPEC = importlib.util.spec_from_file_location("resolve_assistant_popup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_popup_renders_common_markdown_as_html():
    rendered = MODULE.markdown_to_html(
        "**Completed:**\n- `create_timeline`\n[Resolve guide](https://example.com/resolve)"
    )

    assert "<b>Completed:</b>" in rendered
    assert "<ul>" in rendered and "<li><code>create_timeline</code></li>" in rendered
    assert 'href="https://example.com/resolve"' in rendered


def test_popup_escapes_raw_html_before_rendering():
    rendered = MODULE.markdown_to_html("<script>alert('x')</script>")

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
