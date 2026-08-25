"""Fail-fast validation for benchmark tool runtime dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Iterable

from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import Scenario


EEG_RUNTIME_DEPENDENCIES = ("numpy", "scipy", "mne")


def validate_tool_runtime(
    scenarios: Iterable[Scenario], tool_registry: ToolRegistry
) -> None:
    """Raise before evaluation if allowed tools cannot run in this environment."""
    allowed = {
        name
        for scenario in scenarios
        for name in scenario.access.allowed_tools
    }
    if not allowed:
        return

    registered = set(tool_registry.list_names())
    missing_tools = sorted(allowed - registered)
    if missing_tools:
        raise RuntimeError(
            "Scenario tool configuration references unregistered tools: "
            + ", ".join(missing_tools)
        )

    missing_dependencies = []
    for module_name in EEG_RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_dependencies.append(module_name)
    if missing_dependencies:
        raise RuntimeError(
            "EEG tool runtime dependencies are missing: "
            + ", ".join(missing_dependencies)
            + ". Install the project runtime dependencies before evaluation."
        )
