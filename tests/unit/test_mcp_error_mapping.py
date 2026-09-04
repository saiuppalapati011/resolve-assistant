import pytest
from backend.agent.nodes.executor import map_error_message

def test_map_error_message():
    assert map_error_message("DaVinci Resolve error: No project is currently open") == "No project is currently open. Please open a project first."
    assert map_error_message("DaVinci Resolve error: Not connected to DaVinci Resolve") == "Not connected to DaVinci Resolve. Please ensure DaVinci Resolve is running and scripting is enabled."
    assert map_error_message("DaVinci Resolve error: Failed to get Media Pool") == "Could not access the Media Pool."
    assert map_error_message("DaVinci Resolve error: Timeline 'Sequence 1' not found") == "The requested timeline could not be found."
    assert map_error_message("DaVinci Resolve error: No current timeline") == "No timeline is currently active."
    assert map_error_message("DaVinci Resolve error: Failed to get root folder") == "Could not access the root folder."
    
    # Unknown pattern fallback
    assert map_error_message("DaVinci Resolve error: Something weird happened") == "DaVinci Resolve reported an issue: Something weird happened"
    
    # Generic fallback
    assert map_error_message("Normal output") == "Normal output"
