"""Tool loading utilities."""

from __future__ import annotations

from pathlib import Path

from eeg_agent_bench.tools.registry import ToolRegistry


def load_tool_registry(config_path: str | Path | None = None) -> ToolRegistry:
    """Load the tool registry from config.

    If no config path is given or tools list is empty, returns an empty registry.
    """
    if config_path is None:
        return ToolRegistry()
    return ToolRegistry.from_config(config_path)
