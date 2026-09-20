from backend.agent.nodes.executor import _is_current_timeline_alias


def test_executor_accepts_tool_named_current_timeline_aliases():
    assert _is_current_timeline_alias("$get_current_timeline")
    assert _is_current_timeline_alias("__get_current_timeline__")
    assert _is_current_timeline_alias("$current_timeline")


def test_executor_does_not_treat_literal_timeline_name_as_alias():
    assert not _is_current_timeline_alias("Test Main Time")

