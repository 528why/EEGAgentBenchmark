"""Configuration loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ── Agent Config ──────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Agent adapter configuration."""
    name: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    # Direct API key, used only as a fallback when ``api_key_env`` is unset or
    # the env var is empty.  Prefer ``api_key_env`` so secrets stay out of YAML
    # for released configs; ``api_key`` exists for self-hosted gateways.
    api_key: str = ""
    # Authentication style for the gateway:
    #   "bearer"  — standard OpenAI ``Authorization: Bearer <key>`` (default).
    #   "api-key" — send the key in an ``api-key: <key>`` header instead.
    auth_style: str = "bearer"
    # Optional HTTP headers sent on every request for provider-specific APIs.
    extra_headers: dict[str, str] | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    truncate_prompt_tokens: int | None = None
    # Per-request HTTP timeout in seconds for the OpenAI client (fail fast on
    # hung requests; SDK default is 600s).  None → SDK default.
    request_timeout: float | None = None
    # Retry budget.  When set it overrides both the OpenAI SDK's internal
    # retries and this adapter's ``_call_with_retry`` loop (so they don't
    # compound into very long stalls).  None → adapter default (5).
    max_retries: int | None = None
    max_tokens: int = 2048
    response_format: str = "text"  # "text" (free reasoning) | "json_strict" (pure JSON)
    # Arbitrary provider-specific ``extra_body`` payload merged into every
    # chat-completion request (e.g. vLLM/SGLang sampling knobs).  Generic.
    extra_body: dict[str, Any] | None = None
    # Convenience hook for chat-template controls passed via ``extra_body``
    # (e.g. DeepSeek-V3.2 thinking switch: ``{"thinking": true}``).  Merged
    # into ``extra_body["chat_template_kwargs"]`` by the adapter.
    chat_template_kwargs: dict[str, Any] | None = None
    # Tool-calling protocol the runner/adapter should use for this model:
    #   "text"   — tools described in the system prompt; the model emits a
    #              JSON action block in its visible content (provider-agnostic).
    #   "native" — pass tools via the API `tools=` parameter and read the
    #              provider's structured `message.tool_calls` (required by
    #              harmony-format models such as openai/gpt-oss-* which leave
    #              `content` empty on tool-deciding turns).
    tool_protocol: str = "text"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = data.get("agent", data)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def get_api_key(self) -> str:
        """Resolve API key: env var (preferred) → direct ``api_key`` fallback."""
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                return key
            # Env var named but unset — fall back to a direct key if provided,
            # otherwise raise so misconfiguration is loud.
            if self.api_key:
                return self.api_key
            raise OSError(
                f"API key env var '{self.api_key_env}' is not set. "
                f"Please export {self.api_key_env}=<your-key>."
            )
        return self.api_key

    def to_log_dict(self) -> dict[str, Any]:
        """Return a dict safe for logging (no API key)."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
            "max_tokens": self.max_tokens,
            "tool_protocol": self.tool_protocol,
        }


# ── Rollout Config ────────────────────────────────────────────────────

@dataclass
class RolloutConfig:
    """Rollout execution parameters."""
    max_turns: int = 1000
    max_tool_calls: int = 8
    max_tool_calls_per_turn: int = 3
    max_protocol_retries: int = 3
