"""Base agent adapter interface."""

from __future__ import annotations

import abc
from typing import Any

from eeg_agent_bench.config import AgentConfig
from eeg_agent_bench.types import AgentResponse


class AgentAdapter(abc.ABC):
    """Abstract base for agent adapters.

    All models (API or local) expose the same interface to the runner.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tool_protocol(self) -> str:
        """Tool-calling protocol: ``"text"`` (default) or ``"native"``.

        The runner reads this to decide whether to (a) pass tool specs via
        the API ``tools=`` parameter and (b) format the conversation history
        with structured ``tool_calls`` / ``role: "tool"`` messages.
        """
        return getattr(self.config, "tool_protocol", "text") or "text"

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate the next action given conversation history and available tools.

        Args:
            messages: Chat history (system + user + assistant + tool messages).
            tools: OpenAI-format tool specs. Can be empty list or None.
            response_schema: Expected JSON output schema for structured output.

        Returns:
            AgentResponse with parsed action and metadata.
        """
        ...


class AgentAdapterFactory:
    """Factory for creating agent adapters from config."""

    _adapters: dict[str, type[AgentAdapter]] = {}

    @classmethod
    def register(cls, provider: str, adapter_cls: type[AgentAdapter]) -> None:
        cls._adapters[provider] = adapter_cls

    @classmethod
    def create(cls, config: AgentConfig) -> AgentAdapter:
        if config.provider not in cls._adapters:
            raise ValueError(
                f"Unknown agent provider: '{config.provider}'. "
                f"Available: {list(cls._adapters.keys())}"
            )
        return cls._adapters[config.provider](config)
