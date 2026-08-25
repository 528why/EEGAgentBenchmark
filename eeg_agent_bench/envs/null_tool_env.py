"""Null-tool environment — no real tools available.

Tool calls return 'tool_not_available' observations.
Rollout does NOT terminate on tool failure — agent can continue reasoning.
"""

from __future__ import annotations

import json
from typing import Any

from eeg_agent_bench.envs.base import Environment
from eeg_agent_bench.types import AgentAction, Observation, Scenario


class NullToolEnvironment(Environment):
    """Environment with no real tools.

    Any tool call returns a structured 'tool_not_available' observation.
    The agent can still continue reasoning and submit a final answer.
    """

    def __init__(self):
        self._scenario: Scenario | None = None

    def reset(self, scenario: Scenario) -> list[dict[str, Any]]:
        self._scenario = scenario
        # Initial messages are built by the task prompt builder, not the env
        return []

    def step(self, action: AgentAction) -> list[Observation]:
        """All tool calls return tool_not_available."""
        observations: list[Observation] = []
        if action.tool_calls:
            for tc in action.tool_calls:
                observations.append(Observation(
                    status="tool_not_available",
                    content={
                        "error": f"Tool '{tc.name}' is not available in the current environment.",
                        "suggestion": "Please reason based on the clinical context provided and submit your final answer.",
                    },
                    tool_call_id=tc.tool_call_id,
                    error=f"Tool '{tc.name}' is not available.",
                ))
        return observations

    def update_memory(self, action: AgentAction) -> Observation:
        """Acknowledge memory update."""
        return Observation(
            status="ok",
            content={"message": "Memory updated.", "content": action.memory_content},
        )

    def invalid_action(self, action: AgentAction) -> Observation:
        """Return error for invalid actions."""
        return Observation(
            status="invalid_action",
            error="Action type not recognised. Please submit a final_answer, tool_call, or memory_update.",
        )

    def force_finalize(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract best-effort answer from the last assistant message."""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return {"raw_response": content}
        return {"raw_response": "No answer provided."}
