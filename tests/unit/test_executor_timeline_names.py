from backend.agent.nodes.executor import _needs_timeline_name_normalization, _timeline_names_from_result


def test_timeline_tool_names_are_normalized_but_create_is_not():
    assert _needs_timeline_name_normalization("davinci-resolve_switch_timeline")
    assert _needs_timeline_name_normalization("davinci-resolve_get_track_count")
    assert not _needs_timeline_name_normalization("davinci-resolve_create_timeline")


def test_timeline_list_result_supports_mcp_python_list_format():
    assert _timeline_names_from_result("['Timeline 1', 'Test Run']") == ["Timeline 1", "Test Run"]

