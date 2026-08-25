"""OpenAI-compatible agent adapter.

Works with OpenAI API, Azure OpenAI, Claude, Gemini, vLLM, SGLang, TGI,
and any service exposing an OpenAI-compatible chat completion endpoint.

**Text-based tool calling**: tool information is embedded in the system
prompt.  The model outputs a JSON object with ``"action": "tool_call"``
or ``"action": "final_answer"`` in its text response.  No API-level
``tools`` parameter is used — this ensures compatibility across all LLMs.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from eeg_agent_bench.agents.base import AgentAdapter
from eeg_agent_bench.config import AgentConfig
from eeg_agent_bench.types import (
    ActionType,
    AgentAction,
    AgentResponse,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)

# Synthetic terminal tool injected in native function-calling mode so that
# harmony-format models (gpt-oss, ...) can deliver their final answer as a
# structured tool call rather than free-text content (which they often omit).
_SUBMIT_TOOL_NAME = "submit_final_answer"

# Regex to extract JSON from a response that may contain markdown fences
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?requested (\d+) tokens "
    r"\((\d+) in the messages, (\d+) in the completion\)",
    re.DOTALL,
)
# Newer vLLM wording: "maximum context length is 262144 tokens. However, you
# requested 32768 output tokens and your prompt contains at least 229377 input
# tokens, for a total of 262145 tokens."  (group1=max_context, group2=input).
_CONTEXT_LENGTH_RE_V2 = re.compile(
    r"maximum context length is (\d+) tokens.*?"
    r"prompt contains at least (\d+) input tokens",
    re.DOTALL,
)

_TOOL_MARKER_RE = re.compile(
    r"<\|?tool_call(?:_begin)?\|?>|</?tool_call>|\[TOOL_CALLS\]",
    re.IGNORECASE,
)

_TAGGED_TOOL_BLOCK_RE = re.compile(
    r"<\|?tool_call(?:_begin)?\|?>(.*?)</?\|?tool_call(?:_end)?\|?>|"
    r"<tool_call>(.*?)</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TAGGED_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_][\w.-]*)>", re.IGNORECASE,
)
_TAGGED_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _message_text_content(message: Any) -> str:
    """Normalize provider text content, including segmented content arrays."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text") or item.get("content")
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _decode_tagged_parameter(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _extract_tagged_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse ``<tool_call><function=name><parameter=k>v`` model syntax."""
    calls = []
    for match in _TAGGED_TOOL_BLOCK_RE.finditer(text):
        block = next((group for group in match.groups() if group is not None), "")
        function_match = _TAGGED_FUNCTION_RE.search(block)
        if not function_match:
            continue
        arguments = {
            parameter_match.group(1): _decode_tagged_parameter(parameter_match.group(2))
            for parameter_match in _TAGGED_PARAMETER_RE.finditer(block)
        }
        calls.append((function_match.group(1), arguments))
    return calls


def _extract_action_and_thought(text: str) -> tuple[dict | None, str]:
    """Extract a JSON action block and surrounding free-text thought.

    Models naturally produce reasoning text before/after a JSON action.
    This function separates the two:
      - thought: all text outside the JSON block (the model's reasoning)
      - action:  the parsed JSON dict (tool_call / final_answer)

    Returns:
        (action_dict_or_None, thought_text)
    """
    text = text.strip()

    # 1. Entire response is pure JSON (no free text)
    try:
        parsed = json.loads(text)
        # thought might be inside the JSON as a "thought" field
        thought = parsed.pop("thought", "") if isinstance(parsed, dict) else ""
        return parsed, thought
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. JSON inside ```json ... ``` fence
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            # thought = everything outside the fence
            thought = (text[: m.start()] + text[m.end() :]).strip()
            if isinstance(parsed, dict):
                # Also grab any "thought" field inside the JSON
                inner_thought = parsed.pop("thought", "")
                if inner_thought and not thought:
                    thought = inner_thought
            return parsed, thought
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Find first top-level { ... } in free text
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        # thought = text before + text after the JSON
                        thought = (text[:start] + text[i + 1 :]).strip()
                        if isinstance(parsed, dict):
                            inner_thought = parsed.pop("thought", "")
                            if inner_thought and not thought:
                                thought = inner_thought
                        return parsed, thought
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    # 4. No JSON found — entire text is thought (no action)
    return None, text


def _fit_max_tokens_after_context_error(error_text: str) -> int | None:
    """Return a safe max_tokens value for vLLM context-length errors."""
    match = _CONTEXT_LENGTH_RE.search(error_text)
    if match:
        max_context = int(match.group(1))
        prompt_tokens = int(match.group(3))
    else:
        # Fall back to the newer vLLM message format.
        match = _CONTEXT_LENGTH_RE_V2.search(error_text)
        if not match:
            return None
        max_context = int(match.group(1))
        prompt_tokens = int(match.group(2))
    # vLLM enforces prompt_tokens + max_tokens <= max_context. Leave one token
    # of margin because chat templates/tokenizers can vary slightly on retry.
    fitted = max_context - prompt_tokens - 1
    return fitted if fitted >= 1 else None


class OpenAICompatibleAdapter(AgentAdapter):
    """Adapter for OpenAI-compatible chat completion APIs.

    Uses **text-based tool calling** by default: tool specs are in the
    system prompt; the model responds with JSON text containing
    ``"action": "tool_call"`` or ``"action": "final_answer"``.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._client = None
        self._last_error: str = ""  # stash for retry-exhausted errors

    def _get_client(self):
        """Lazy-init the OpenAI client.

        Supports two auth conventions:
          * ``auth_style: bearer`` (default) — the SDK sends the standard
            ``Authorization: Bearer <key>`` header.
          * ``auth_style: api-key`` — additionally inject an ``api-key: <key>``
            header for services that use that authentication convention.
        Any configured ``extra_headers`` are merged into ``default_headers``
        and sent on every request.
        """
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")
            api_key = self.config.get_api_key()
            default_headers: dict[str, str] = {}
            extra = getattr(self.config, "extra_headers", None)
            if extra:
                default_headers.update(extra)
            if getattr(self.config, "auth_style", "bearer") == "api-key":
                default_headers["api-key"] = api_key
            client_kwargs: dict[str, Any] = dict(
                api_key=api_key,
                base_url=self.config.base_url or None,
                default_headers=default_headers or None,
            )
            # Optional per-request timeout (seconds) — fail fast on hung
            # requests instead of the SDK's 600s default (e.g. thinking models
            # on very large prompts).  Only applied when set in config.
            rt = getattr(self.config, "request_timeout", None)
            if rt is not None:
                client_kwargs["timeout"] = rt
            # Optional SDK-level retry count.  When set (e.g. 0), disables the
            # OpenAI client's own internal retries so they don't compound with
            # this adapter's ``_call_with_retry`` backoff loop.
            mr = getattr(self.config, "max_retries", None)
            if mr is not None:
                client_kwargs["max_retries"] = mr
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _effective_max_retries(self) -> int:
        """Adapter-level retry budget; overridable via config ``max_retries``."""
        mr = getattr(self.config, "max_retries", None)
        return mr if mr is not None else self._MAX_RETRIES

    # Retry settings for transient errors (5xx, network, rate-limit).
    _MAX_RETRIES = 5
    _RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt
    _RETRY_MAX_DELAY = 120.0  # cap
    # Free-tier providers intermittently return HTTP 200 with an empty body
    # (``choices`` is None / []), especially after a large tool result.
    # Retry these with the explicit backoff below before giving up.
    _EMPTY_RETRY_DELAYS = (5.0, 15.0, 30.0)

    @staticmethod
    def _has_usable_choice(response: Any) -> bool:
        """True if the API response carries at least one choice with a message."""
        if response is None:
            return False
        choices = getattr(response, "choices", None)
        if not choices:
            return False
        try:
            return choices[0].message is not None
        except (IndexError, AttributeError, TypeError):
            return False

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the exception looks like a transient / server error."""
        text = str(exc)
        # HTTP 5xx
        for code in ("500", "502", "503", "521", "522", "524"):
            if f"Error code: {code}" in text or f"HTTP/1.1 {code}" in text:
                return True
        # Explicit server-error keywords
        for kw in ("server is down", "origin_down", "Connection", "Timeout",
                    "timed out", "reset by peer", "SSLError", "RemoteDisconnected"):
            if kw.lower() in text.lower():
                return True
        # Rate-limit (429)
        if "Error code: 429" in text or "rate" in text.lower():
            return True
        # Empty / non-JSON body: some gateways return HTTP 200 with an empty
        # body (esp. on large requests), which the SDK surfaces as a JSON
        # decode error ("Expecting value: line 1 column 1 (char 0)").  Treat
        # as transient and retry with backoff instead of failing the scenario.
        low = text.lower()
        for kw in ("expecting value", "jsondecodeerror",
                   "line 1 column 1", "unexpected end of data",
                   "unexpected mimetype", "unexpected end of json"):
            if kw in low:
                return True
        return False

    def _call_with_retry(self, client, kwargs: dict[str, Any]):
        """Call the API with exponential-backoff retry on transient errors.

        Also handles context-length errors by shrinking max_tokens.
        Returns the response object, or ``None`` if all retries exhausted
        (the error text is stashed in ``self._last_error``).
        """
        delay = self._RETRY_BASE_DELAY
        last_exc: Exception | None = None
        max_retries = self._effective_max_retries()

        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                last_exc = e
                error_text = str(e)

                # Context-length error — shrink max_tokens, retry once
                fitted = _fit_max_tokens_after_context_error(error_text)
                if fitted is not None and fitted < kwargs["max_tokens"]:
                    logger.warning(
                        "Context window exceeded; retrying with "
                        "max_tokens=%s (was %s).",
                        fitted, kwargs["max_tokens"],
                    )
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = fitted
                    try:
                        return client.chat.completions.create(**kwargs)
                    except Exception as retry_err:
                        self._last_error = str(retry_err)
                        logger.error("Failed after context-length retry: %s", retry_err)
                        return None

                # Transient / server error — back off and retry
                if self._is_retryable(e) and attempt < max_retries:
                    logger.warning(
                        "Transient error (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, max_retries, delay,
                        error_text[:120],
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self._RETRY_MAX_DELAY)
                    continue

                # Non-retryable or retries exhausted
                self._last_error = error_text
                logger.error("Agent generation failed: %s", error_text[:200])
                return None

        # Should not reach here, but safety net
        self._last_error = str(last_exc) if last_exc else "unknown error"
        return None

    def _extra_body_overrides(self) -> dict[str, Any]:
        """Provider-/model-specific ``extra_body`` payload.

        Base behaviour: merge the config's generic ``extra_body`` dict and the
        ``chat_template_kwargs`` convenience field.  Subclasses (e.g. the
        DeepSeek-V3.2 adapter) override this to inject their own defaults.
        """
        out: dict[str, Any] = {}
        cfg_extra = getattr(self.config, "extra_body", None)
        if cfg_extra:
            out.update(cfg_extra)
        ctk = getattr(self.config, "chat_template_kwargs", None)
        if ctk:
            existing = out.get("chat_template_kwargs") or {}
            out["chat_template_kwargs"] = {**existing, **ctk}
        return out

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AgentResponse:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        extra_body: dict[str, Any] = {}
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            extra_body["top_k"] = self.config.top_k
        if self.config.min_p is not None:
            extra_body["min_p"] = self.config.min_p
        if self.config.truncate_prompt_tokens is not None:
            extra_body["truncate_prompt_tokens"] = self.config.truncate_prompt_tokens
        # Provider-/model-specific extra_body (generic dict) + chat-template
        # controls (e.g. DeepSeek-V3.2 thinking switch).  Subclasses may inject
        # further defaults via ``_extra_body_overrides``.
        for k, v in self._extra_body_overrides().items():
            extra_body[k] = v
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Text-based tool calling: tool specs live in the system prompt.
        # Do NOT force response_format=json — the model should be free
        # to output reasoning text around the JSON action block.
        # Only set response_format if explicitly configured as "json_strict".
        if self.config.response_format == "json_strict":
            kwargs["response_format"] = {"type": "json_object"}

        # ── Native function-calling protocol ──────────────────────
        # For harmony-format models (e.g. openai/gpt-oss-*) that emit
        # actions via the structured ``tool_calls`` channel and leave
        # ``content`` empty, pass the tool specs through the API ``tools``
        # parameter.  We also append a synthetic ``submit_final_answer``
        # function so the model can deliver its final answer as a
        # structured tool call — these models reliably populate tool-call
        # arguments even when they refuse to emit free-text content.
        native = self.tool_protocol == "native"
        if native:
            # Always expose the synthetic ``submit_final_answer`` tool so a
            # harmony model can deliver its answer as a structured tool call
            # even for **no-tool** tasks (e.g. K0 knowledge QA), where
            # ``tools`` is empty and these models would otherwise leave
            # ``content`` empty and submit nothing.
            kwargs["tools"] = list(tools or []) + [self._build_submit_tool(response_schema)]
            kwargs["tool_choice"] = "auto"

        t0 = time.monotonic()
        # Outer loop: guard against empty (no-choices) responses from
        # free-tier providers.  Each attempt itself uses _call_with_retry
        # for transient network/5xx/429 errors.
        response = self._call_with_retry(client, kwargs)
        for delay in self._EMPTY_RETRY_DELAYS:
            if response is None or self._has_usable_choice(response):
                break
            logger.warning(
                "Empty response (no choices); retrying in %.0fs.", delay,
            )
            time.sleep(delay)
            response = self._call_with_retry(client, kwargs)

        if response is None:
            # All retries exhausted — already logged inside _call_with_retry
            latency_ms = (time.monotonic() - t0) * 1000
            return AgentResponse(
                action=AgentAction(action_type=ActionType.FINAL_ANSWER),
                error=self._last_error,
                latency_ms=latency_ms,
            )
        if not self._has_usable_choice(response):
            # Empty response persisted after retries — fail this scenario
            # cleanly instead of crashing on response.choices[0].
            latency_ms = (time.monotonic() - t0) * 1000
            return AgentResponse(
                action=AgentAction(action_type=ActionType.FINAL_ANSWER),
                error="API returned an empty response (no choices) after retries.",
                latency_ms=latency_ms,
            )
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        message = choice.message

        # Parse token usage
        token_usage = {}
        if response.usage:
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # OpenRouter / some providers expose reasoning_tokens in
            # usage.completion_tokens_details.reasoning_tokens.
            details = getattr(response.usage, "completion_tokens_details", None)
            if details is not None:
                rt = getattr(details, "reasoning_tokens", None)
                if rt is not None:
                    token_usage["reasoning_tokens"] = rt

        # Parse action from text response (or native tool_calls)
        action = self._parse_action(message, native=native)

        # ── Extract provider-level reasoning (OpenRouter, etc.) ────
        # Some providers return chain-of-thought in a separate
        # ``message.reasoning`` field rather than inside ``content``.
        # DeepSeek-style models expose chain-of-thought in ``reasoning_content``;
        # OpenRouter and others use ``reasoning``.  Accept either.
        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )
        if not reasoning:
            # Fallback: try model_dump() dict for providers that
            # expose it as a plain dict key but not an attribute.
            msg_dict = (
                message.model_dump()
                if hasattr(message, "model_dump")
                else {}
            )
            if isinstance(msg_dict, dict):
                reasoning = msg_dict.get("reasoning") or msg_dict.get(
                    "reasoning_content"
                )
        if reasoning:
            action.reasoning = reasoning

        return AgentResponse(
            action=action,
            raw_response=message.model_dump() if hasattr(message, "model_dump") else str(message),
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _build_submit_tool(response_schema: dict[str, Any] | None) -> dict[str, Any]:
        """Build the synthetic ``submit_final_answer`` tool spec.

        Its parameters mirror the task's expected output schema so the model
        returns the answer's required fields directly in the tool-call
        arguments.  Falls back to a permissive object schema if none given.
        """
        if isinstance(response_schema, dict) and response_schema.get("properties"):
            params = response_schema
        else:
            params = {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        return {
            "type": "function",
            "function": {
                "name": _SUBMIT_TOOL_NAME,
                "description": (
                    "Submit your FINAL answer for the task once your analysis "
                    "is complete. Provide all required fields of the task's "
                    "output schema. Call this exactly once to finish."
                ),
                "parameters": params,
            },
        }

    def _parse_action(self, message: Any, native: bool = False) -> AgentAction:
        """Parse the assistant message into an AgentAction.

        Two protocols are supported:

        - **native** (``tool_protocol: native``): the action arrives in the
          provider's structured ``message.tool_calls``; ``content`` may be
          empty.  Used by harmony-format models (openai/gpt-oss-*).
        - **text** (default): the model emits a JSON action block inside its
          visible ``content``; this method separates the free-text *thought*
          from the JSON *action*.
        """
        content = _message_text_content(message)
        raw = message.model_dump() if hasattr(message, "model_dump") else None

        # ── Native function-calling: read structured tool_calls ───
        if native:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                calls: list[ToolCallRequest] = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "") if fn else ""
                    raw_args = getattr(fn, "arguments", "") if fn else ""
                    try:
                        arguments = json.loads(raw_args) if raw_args else {}
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    # ``submit_final_answer`` is the synthetic terminal tool:
                    # its arguments ARE the structured final answer.
                    if name == _SUBMIT_TOOL_NAME:
                        return AgentAction(
                            action_type=ActionType.FINAL_ANSWER,
                            final_answer=arguments,
                            raw_message=raw,
                            thought=content,
                        )
                    calls.append(ToolCallRequest(
                        # Use the provider's id so the follow-up tool message
                        # can reference it (required by the OpenAI tool role).
                        tool_call_id=getattr(tc, "id", None) or str(uuid.uuid4()),
                        name=name,
                        arguments=arguments,
                    ))
                if calls:
                    return AgentAction(
                        action_type=ActionType.TOOL_CALL,
                        tool_calls=calls,
                        raw_message=raw,
                        thought=content,
                    )
            # No tool_calls → the model is answering directly; the final
            # answer is in ``content`` (parsed below by the shared path).

        tagged_calls = _extract_tagged_tool_calls(content)
        if tagged_calls:
            return AgentAction(
                action_type=ActionType.TOOL_CALL,
                tool_calls=[
                    ToolCallRequest(
                        tool_call_id=str(uuid.uuid4()),
                        name=name,
                        arguments=arguments,
                    )
                    for name, arguments in tagged_calls
                ],
                raw_message=raw,
                thought="",
            )

        parsed, thought = _extract_action_and_thought(content)

        if parsed is None:
            # No JSON action found. This is NOT a final answer: treating a
            # reasoning-only or empty turn as terminal caused large-scale
            # zero-coverage outputs on long-horizon tasks. The runner will
            # issue a bounded protocol-repair nudge.
            return AgentAction(
                action_type=ActionType.INVALID,
                raw_message=raw,
                thought=thought or content,
            )

        # ── Top-level JSON array ──────────────────────────────────
        # Some models (e.g. GLM-5.1) occasionally emit a JSON *array* instead
        # of a single object — typically a batch of tool calls, or the answer
        # wrapped in a list.  Without this guard ``parsed.get(...)`` below would
        # raise ``'list' object has no attribute 'get'`` and fail the scenario.
        if isinstance(parsed, list):
            tool_calls: list[ToolCallRequest] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool_name") or item.get("name")
                if item.get("action") == "tool_call" or name:
                    if name:
                        args = item.get("arguments", {})
                        tool_calls.append(ToolCallRequest(
                            tool_call_id=str(uuid.uuid4()),
                            name=name,
                            arguments=args if isinstance(args, dict) else {},
                        ))
            if tool_calls:
                return AgentAction(
                    action_type=ActionType.TOOL_CALL,
                    tool_calls=tool_calls,
                    raw_message=raw,
                    thought=thought,
                )
            # Otherwise fall back to the first dict element (if any) as the
            # action object; else treat the whole text as a final answer.
            dicts = [x for x in parsed if isinstance(x, dict)]
            if dicts:
                parsed = dicts[0]
            else:
                return AgentAction(
                    action_type=ActionType.INVALID,
                    raw_message=raw,
                    thought=thought or content,
                )

        if not isinstance(parsed, dict):
            # Scalar / unexpected JSON (number, string, bool) — no action.
            return AgentAction(
                action_type=ActionType.INVALID,
                raw_message=raw,
                thought=thought or content,
            )

        action_field = parsed.get("action", "")

        # ── tool_call ─────────────────────────────────────────────
        if action_field == "tool_call":
            arguments = parsed.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            tool_name = parsed.get("tool_name", "") or arguments.pop("tool_name", "")
            if not tool_name:
                return AgentAction(
                    action_type=ActionType.INVALID,
                    raw_message=raw,
                    thought=thought or content,
                )
            return AgentAction(
                action_type=ActionType.TOOL_CALL,
                tool_calls=[ToolCallRequest(
                    tool_call_id=str(uuid.uuid4()),
                    name=tool_name,
                    arguments=arguments,
                )],
                raw_message=raw,
                thought=thought,
            )

        # ── memory_update ─────────────────────────────────────────
        if action_field == "memory_update":
            return AgentAction(
                action_type=ActionType.MEMORY_UPDATE,
                memory_content=parsed.get("memory", parsed.get("content", "")),
                raw_message=raw,
                thought=thought,
            )

        # ── final_answer (explicit or implicit) ───────────────────
        if action_field == "final_answer" or "classification" in parsed:
            return AgentAction(
                action_type=ActionType.FINAL_ANSWER,
                final_answer=parsed,
                raw_message=raw,
                thought=thought,
            )

        # Some model-specific tool syntaxes wrap a bare argument object in a
        # tool marker. Recover it when a tool name is present rather than
        # silently terminating the rollout.
        if _TOOL_MARKER_RE.search(content):
            arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else parsed
            tool_name = parsed.get("tool_name") or parsed.get("name")
            if not tool_name and isinstance(arguments, dict):
                tool_name = arguments.get("tool_name") or arguments.get("name")
                arguments = dict(arguments)
                arguments.pop("tool_name", None)
                arguments.pop("name", None)
            if tool_name:
                return AgentAction(
                    action_type=ActionType.TOOL_CALL,
                    tool_calls=[ToolCallRequest(
                        tool_call_id=str(uuid.uuid4()),
                        name=str(tool_name),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )],
                    raw_message=raw,
                    thought=thought,
                )

        # A JSON object without an explicit action is ambiguous. Do not treat
        # intermediate tool arguments or analysis dictionaries as terminal.
        # The runner will request an explicit action and validate final schema.
        return AgentAction(
            action_type=ActionType.INVALID,
            raw_message=raw,
            thought=thought or content,
        )
