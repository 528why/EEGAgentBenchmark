"""Tool registry — manages available tools for scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eeg_agent_bench.core.registry import Registry
from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import Scenario, ToolSpec


class ToolRegistry:
    """Manages tool registration and lookup.

    First version: returns empty tool list.
    Future: loads tool specs from config, registers implementations.
    """

    def __init__(self):
        self._registry: Registry[BaseTool] = Registry("tool")
        self._specs: dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool implementation."""
        self._registry.register(tool.spec.name, tool)
        self._specs[tool.spec.name] = tool.spec

    def get(self, name: str) -> BaseTool:
        return self._registry.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def list_for_scenario(self, scenario: Scenario) -> list[dict[str, Any]]:
        """Return OpenAI-format tool specs allowed for a scenario.

        Args:
            scenario: the scenario whose ``access.allowed_tools`` gates the set.
        """
        allowed = scenario.access.allowed_tools
        if not allowed:
            return []
        return [
            self._specs[name].to_openai_tool()
            for name in allowed
            if name in self._specs
        ]

    def list_names(self) -> list[str]:
        return self._registry.list_names()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ToolRegistry":
        """Load tool registry from config file and register implementations."""
        registry = cls()
        config_path = Path(config_path)
        if not config_path.exists():
            return registry

        with open(config_path) as f:
            data = yaml.safe_load(f)

        enabled_names = data.get("tools", [])
        if not enabled_names:
            return registry

        # Import all available tool implementations
        try:
            from eeg_agent_bench.tools.implementations import ALL_TOOLS
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "Could not import tool implementations. "
                "Tools will not be available."
            )
            return registry

        # Build name → class lookup
        tool_classes = {cls_().spec.name: cls_ for cls_ in ALL_TOOLS}

        for name in enabled_names:
            if name in tool_classes:
                registry.register(tool_classes[name]())

        return registry
