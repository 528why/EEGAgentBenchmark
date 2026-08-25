"""Base tool interface."""

from __future__ import annotations

import abc
from typing import Any

from eeg_agent_bench.types import ToolSpec


class BaseTool(abc.ABC):
    """Abstract base for EEG analysis tools."""

    @property
    @abc.abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        ...

    @abc.abstractmethod
    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given arguments.

        Args:
            arguments: Tool input parameters.
            context: Runtime context (record paths, workspace, etc.)
                     Must NOT contain gold/answer key data.

        Returns:
            Structured output matching the tool's output_schema.
        """
        ...
