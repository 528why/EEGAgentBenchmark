"""Base environment interface."""

from __future__ import annotations

import abc
from typing import Any

from eeg_agent_bench.types import AgentAction, Observation, Scenario


class Environment(abc.ABC):
    """Abstract base for agent execution environments.

    Responsibilities:
    - reset: set up scenario, return initial messages
    - step: execute tool calls, return observations
    - update_memory: handle agent memory updates
    - invalid_action: handle unrecognised actions
    - force_finalize: extract answer when max_turns exceeded
    """

    @abc.abstractmethod
    def reset(self, scenario: Scenario) -> list[dict[str, Any]]:
        """Initialize environment for a scenario. Returns initial prompt messages."""
        ...

    @abc.abstractmethod
    def step(self, action: AgentAction) -> list[Observation]:
        """Execute tool calls in the action. Returns observations."""
        ...

    @abc.abstractmethod
    def update_memory(self, action: AgentAction) -> Observation:
        """Handle agent memory update. Returns acknowledgement."""
        ...

    @abc.abstractmethod
    def invalid_action(self, action: AgentAction) -> Observation:
        """Handle invalid/unrecognised action. Returns error observation."""
        ...

    @abc.abstractmethod
    def force_finalize(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract best-effort answer when max_turns is exceeded."""
        ...
