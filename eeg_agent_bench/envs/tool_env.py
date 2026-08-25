"""Tool environment — executes the deterministic EEG analysis tools."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from eeg_agent_bench.envs.base import Environment
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import AgentAction, Observation, Scenario

logger = logging.getLogger(__name__)


class ToolEnvironment(Environment):
    """Environment with real tool execution.

    Validates tool names, arguments, and permissions before execution.
    Wraps errors into structured observations — never crashes the runner.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        timeout_sec: int = 60,
    ):
        self._registry = tool_registry
        self._scenario: Scenario | None = None
        self._timeout_sec = timeout_sec
        self._context: dict[str, Any] = {}
        self._step_counter = 0

    def reset(self, scenario: Scenario) -> list[dict[str, Any]]:
        self._scenario = scenario
        self._step_counter = 0
        # Build context that tools can use to locate data.
        # record_paths maps record_id → file path so tools can load data
        # without the agent having to pass raw paths as arguments.
        record_paths: dict[str, str] = {}
        record_metadata: dict[str, dict] = {}
        for r in scenario.records:
            if r.data_path:
                record_paths[r.record_id] = r.data_path
            if r.metadata:
                record_metadata[r.record_id] = r.metadata

        self._context = {
            "scenario_id": scenario.scenario_id,
            "dataset": scenario.dataset,
            "records": [r.record_id for r in scenario.records],
            "record_paths": record_paths,
            "record_metadata": record_metadata,
        }

        return []

    def step(self, action: AgentAction) -> list[Observation]:
        observations: list[Observation] = []
        if not action.tool_calls:
            return observations

        allowed = set(self._scenario.access.allowed_tools) if self._scenario else set()

        for tc in action.tool_calls:
            # Permission check
            if tc.name not in allowed:
                observations.append(Observation(
                    status="tool_not_available",
                    tool_call_id=tc.tool_call_id,
                    error=f"Tool '{tc.name}' is not allowed for this scenario.",
                ))
                continue

            # Registry check
            if tc.name not in self._registry.list_names():
                observations.append(Observation(
                    status="tool_not_available",
                    tool_call_id=tc.tool_call_id,
                    error=f"Tool '{tc.name}' is not registered.",
                ))
                continue

            # Execute
            tool = self._registry.get(tc.name)
            self._step_counter += 1
            self._context["step"] = self._step_counter
            t0 = time.monotonic()
            try:
                result = tool.execute(tc.arguments, self._context)
                latency = (time.monotonic() - t0) * 1000

                observations.append(Observation(
                    status="ok",
                    content=result,
                    tool_call_id=tc.tool_call_id,
                    latency_ms=latency,
                ))
            except Exception as e:
                latency = (time.monotonic() - t0) * 1000
                observations.append(Observation(
                    status="error",
                    tool_call_id=tc.tool_call_id,
                    latency_ms=latency,
                    error=f"Tool execution failed: {e}",
                ))

        return observations

    def update_memory(self, action: AgentAction) -> Observation:
        return Observation(
            status="ok",
            content={"message": "Memory updated.", "content": action.memory_content},
        )

    def invalid_action(self, action: AgentAction) -> Observation:
        return Observation(
            status="invalid_action",
            error="Action type not recognised.",
        )

    def force_finalize(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return {"raw_response": content}
        return {"raw_response": "No answer provided."}
