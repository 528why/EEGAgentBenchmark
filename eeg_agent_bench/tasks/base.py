"""Base task interface."""

from __future__ import annotations

import abc
from typing import Any

from eeg_agent_bench.types import Scenario


class BaseTask(abc.ABC):
    """Abstract base for task-specific logic."""

    @property
    @abc.abstractmethod
    def task_id(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def task_name(self) -> str:
        ...

    @abc.abstractmethod
    def build_messages(
        self,
        scenario: Scenario,
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the initial prompt messages for this task."""
        ...

    @abc.abstractmethod
    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        """Parse the agent's final answer into structured output."""
        ...
