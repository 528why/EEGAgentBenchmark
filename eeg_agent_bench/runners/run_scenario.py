"""Single scenario runner — the core agent-environment interaction loop.

Implements the multi-turn rollout protocol from Section 6 of the architecture doc.
All tasks (including T1) run through this loop, even when tools==[].
"""

from __future__ import annotations

import logging
import time
from typing import Any

from eeg_agent_bench.agents.base import AgentAdapter
from eeg_agent_bench.config import RolloutConfig
from eeg_agent_bench.envs.base import Environment
from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import (
    ActionType,
    AgentAction,
    EvalResult,
    EvalScore,
    RolloutState,
    Scenario,
    TrajectoryEntry,
    TrajectoryEventType,
)

logger = logging.getLogger(__name__)

# Native-protocol empty-turn recovery (harmony models, e.g. gpt-oss).
_MAX_EMPTY_NUDGES = 2
_EMPTY_ANSWER_NUDGE = (
    "You ended your previous turn without any content or tool call. "
    "If your analysis is complete, call the `submit_final_answer` tool now "
    "with all required fields of the task's output schema. Otherwise, call "
    "another analysis tool to gather the evidence you still need."
)

_TEXT_ACTION_NUDGE = (
    "Your previous turn did not contain a valid action. Continue the task by "
    "responding with exactly one explicit JSON action: either "
    '{"action":"tool_call","tool_name":"<allowed tool>","arguments":{...}} '
    "or a final answer whose fields satisfy the required output schema. "
    "Do not return tool arguments without action/tool_name, and do not leave "
    "the visible content empty."
)

_TOOL_BUDGET_NUDGE = (
    "The tool-call budget is exhausted. Do not call another tool. Submit your "
    "best final answer now with every required output-schema field."
)


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$" ) -> list[str]:
    """Validate the small JSON-Schema subset used by benchmark outputs."""
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected = schema.get("type")
    type_ok = True
    if expected == "object":
        type_ok = isinstance(value, dict)
    elif expected == "array":
        type_ok = isinstance(value, list)
    elif expected == "string":
        type_ok = isinstance(value, str)
    elif expected == "integer":
        type_ok = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "boolean":
        type_ok = isinstance(value, bool)
    if expected and not type_ok:
        return [f"{path}: expected {expected}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(_schema_errors(value[key], child_schema, f"{path}.{key}"))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
    return errors


def _sleep_hypnogram_errors(answer: dict[str, Any], scenario: Scenario) -> list[str]:
    if scenario.task_name != "sleep_staging":
        return []
    rle = answer.get("hypnogram_rle")
    if not isinstance(rle, list) or not rle:
        return ["$.hypnogram_rle: must contain at least one segment"]
    n_epochs = None
    if scenario.records:
        n_epochs = scenario.records[0].metadata.get("n_epochs_total")
    if n_epochs is None:
        n_epochs = (scenario.input.get("metadata") or {}).get("n_epochs_total")
    errors: list[str] = []
    expected_start = 0
    for index, segment in enumerate(rle):
        if not isinstance(segment, dict):
            continue
        start = segment.get("start_epoch")
        end = segment.get("end_epoch")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start != expected_start:
            errors.append(
                f"$.hypnogram_rle[{index}]: expected start_epoch {expected_start}, got {start}"
            )
        if end < start:
            errors.append(f"$.hypnogram_rle[{index}]: end_epoch precedes start_epoch")
        expected_start = end + 1
    if isinstance(n_epochs, int) and expected_start != n_epochs:
        errors.append(
            f"$.hypnogram_rle: must cover epochs 0..{n_epochs - 1}; ended at {expected_start - 1}"
        )
    return errors


def _final_answer_errors(
    answer: Any, schema: dict[str, Any], scenario: Scenario
) -> list[str]:
    if not isinstance(answer, dict):
        return ["$: final answer must be a JSON object"]
    errors = _schema_errors(answer, schema or {})
    if not errors:
        errors.extend(_sleep_hypnogram_errors(answer, scenario))
    return errors


def _repair_nudge(use_native: bool, errors: list[str] | None = None) -> str:
    if errors:
        detail = "; ".join(errors[:6])
        if use_native:
            return (
                "Your submitted final answer did not satisfy the required "
                f"schema: {detail}. Call `submit_final_answer` again with all "
                "required fields and valid value types."
            )
        return (
            "Your submitted final answer did not satisfy the required output "
            f"schema: {detail}. Respond with exactly one corrected "
            '{"action":"final_answer", ...} JSON object containing every '
            "required field."
        )
    return _EMPTY_ANSWER_NUDGE if use_native else _TEXT_ACTION_NUDGE


def _is_empty_final(action: Any) -> bool:
    """True if a final-answer action carries no usable answer payload."""
    fa = action.final_answer or {}
    if not fa:
        return True
    # Adapter emits {"raw_response": ""} when content was empty.
    if set(fa.keys()) <= {"raw_response", "parse_error"}:
        return not (fa.get("raw_response") or "").strip()
    return False


def _native_assistant_message(action: Any) -> dict[str, Any]:
    """Build an OpenAI-format assistant message for the *native* protocol.

    Carries structured ``tool_calls`` (so the provider can match the
    follow-up ``role: "tool"`` results).  ``content`` is the model's
    visible text (often empty on tool-deciding turns for harmony models).
    """
    import json as _json

    content = ""
    if action.raw_message and isinstance(action.raw_message, dict):
        content = action.raw_message.get("content") or ""

    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if action.has_tool_calls and action.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.tool_call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": _json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in action.tool_calls
        ]
    return msg


def _native_tool_messages(obs: Any) -> list[dict[str, Any]]:
    """Build a ``role: "tool"`` result message for the native protocol."""
    content_str = str(obs.content) if obs.content else (obs.error or "")
    msgs: list[dict[str, Any]] = [{
        "role": "tool",
        "tool_call_id": obs.tool_call_id,
        "content": content_str,
    }]
    return msgs


def run_scenario(
    scenario: Scenario,
    agent: AgentAdapter,
    env: Environment,
    tool_registry: ToolRegistry,
    evaluators: list[BaseEvaluator],
    initial_messages: list[dict[str, Any]],
    rollout_config: RolloutConfig | None = None,
) -> tuple[EvalResult, list[TrajectoryEntry]]:
    """Run a single scenario through the multi-turn agent-env loop.

    Returns:
        (EvalResult, trajectory_entries)
    """
    config = rollout_config or RolloutConfig()
    trajectory: list[TrajectoryEntry] = []
    step_counter = 0

    # ── Merge turn limit ──────────────────────────────────────────
    # Scenario-level max_turns may differ from the global RolloutConfig
    # (e.g. T6 sequential tasks need more turns).  Use the smaller of
    # the two so that the scenario cannot exceed the global budget, but
    # tighter per-scenario limits are respected.
    effective_max_turns = min(config.max_turns, scenario.access.max_turns)

    # ── Reset environment ──────────────────────────────────────────
    env.reset(scenario)

    # ── Build initial history ──────────────────────────────────────
    history = list(initial_messages)
    tools = tool_registry.list_for_scenario(scenario)
    # Tool-calling protocol (text vs native function-calling).  Native
    # models receive tool specs via the API ``tools=`` param and require
    # structured ``tool_calls`` / ``role: "tool"`` history messages.
    use_native = getattr(agent, "tool_protocol", "text") == "native"

    # Log prompt
    trajectory.append(TrajectoryEntry(
        scenario_id=scenario.scenario_id,
        step=step_counter,
        event_type=TrajectoryEventType.PROMPT,
        agent_message={"messages": history},
    ))

    state = RolloutState.RUNNING
    prediction: dict[str, Any] = {}
    total_tool_calls = 0
    empty_nudges = 0  # native-protocol empty-turn recovery counter
    protocol_retries = 0
    consecutive_protocol_failures = 0
    format_failure = False
    t_start = time.monotonic()

    # ── Cumulative token counters ─────────────────────────────────
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_tokens = 0

    # ── Multi-turn rollout loop ────────────────────────────────────
    for turn in range(effective_max_turns):
        step_counter += 1

        # Agent generates next action.  In *text* protocol the tool specs
        # already live in the system prompt, so we pass tools=None.  In
        # *native* protocol we hand the specs to the API ``tools=`` param.
        try:
            response = agent.generate(
                messages=history,
                tools=tools if use_native else None,
                response_schema=scenario.expected_output_schema,
            )
        except Exception as e:
            logger.error(f"[{scenario.scenario_id}] Agent error at turn {turn}: {e}")
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.ERROR,
                error=str(e),
            ))
            state = RolloutState.FAILED
            break

        if response.error:
            logger.warning(f"[{scenario.scenario_id}] Agent returned error: {response.error}")
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.ERROR,
                error=response.error,
            ))
            state = RolloutState.FAILED
            break

        action = response.action

        # ── Accumulate token usage ────────────────────────────────
        # NB: use ``(... or 0)`` not ``.get(k, 0)`` — some gateways/adapters
        # (e.g. Gemini/Anthropic native) return the key present but ``None``
        # when usage metadata omits a field; ``.get(k, 0)`` would then yield
        # ``None`` and crash the whole scenario on ``int += None``.
        tu = response.token_usage
        total_prompt_tokens += tu.get("prompt_tokens") or 0
        total_completion_tokens += tu.get("completion_tokens") or 0
        total_reasoning_tokens += tu.get("reasoning_tokens") or 0
        total_tokens += tu.get("total_tokens") or 0

        # Log agent action (includes thought + full model output)
        agent_msg: dict[str, Any] = {
            "action_type": action.action_type.value,
            "thought": action.thought,
            "token_usage": response.token_usage,
            "latency_ms": response.latency_ms,
        }
        # Preserve the model's complete text output so that reasoning
        # traces (including <think> blocks, ReAct chains, etc.) are
        # available for post-hoc analysis and debugging.
        if action.raw_message and isinstance(action.raw_message, dict):
            agent_msg["raw_content"] = action.raw_message.get("content", "")
        # Provider-level reasoning (e.g. OpenRouter message.reasoning)
        # is stored separately from content-extracted thought.
        if action.reasoning:
            agent_msg["reasoning"] = action.reasoning
        trajectory.append(TrajectoryEntry(
            scenario_id=scenario.scenario_id,
            step=step_counter,
            event_type=TrajectoryEventType.AGENT_ACTION,
            agent_message=agent_msg,
        ))

        # ── Append agent's response to history ───────────────────
        # The model must see its own previous thoughts + actions in
        # the conversation history to maintain a coherent reasoning
        # chain (ReAct pattern).
        if use_native:
            # Native protocol: the assistant turn must carry structured
            # ``tool_calls`` so the provider can match the follow-up
            # ``role: "tool"`` results.  Content may be empty.
            history.append(_native_assistant_message(action))
        elif action.raw_message and isinstance(action.raw_message, dict):
            # Text protocol: replay the model's own visible content.  Some
            # reasoning models (e.g. kimi-k3) occasionally emit an EMPTY
            # ``content`` and route everything to ``reasoning_content``.
            # Some strict OpenAI-compatible providers reject an empty assistant
            # message on the next request ("the message at
            # position N with role 'assistant' must not be empty"), which
            # kills the scenario.  Fall back to the parsed thought/reasoning
            # (or a placeholder) so the replayed turn is never empty.
            replay_content = action.raw_message.get("content") or ""
            if not replay_content.strip():
                replay_content = (
                    (action.thought or "").strip()
                    or (action.reasoning or "").strip()
                    or "(no output)"
                )
            history.append({
                "role": "assistant",
                "content": replay_content,
            })
        elif action.thought or action.final_answer or action.has_tool_calls:
            # Reconstruct the assistant message from parsed action
            import json as _json
            assistant_content: dict[str, Any] = {}
            if action.thought:
                assistant_content["thought"] = action.thought
            assistant_content["action"] = action.action_type.value
            if action.has_tool_calls and action.tool_calls:
                assistant_content["tool_name"] = action.tool_calls[0].name
                assistant_content["arguments"] = action.tool_calls[0].arguments
            if action.is_final_answer and action.final_answer:
                assistant_content.update(action.final_answer)
            history.append({
                "role": "assistant",
                "content": _json.dumps(assistant_content, ensure_ascii=False),
            })

        # ── Final answer ──────────────────────────────────────────
        if action.is_final_answer:
            # Native-protocol empty-turn recovery: harmony models
            # (gpt-oss) sometimes end a turn with empty content AND no
            # tool_calls — i.e. they "stop after reasoning".  Rather than
            # accept an empty prediction, nudge the model to emit its
            # answer in the content channel and continue (bounded retry).
            if (use_native and _is_empty_final(action)
                    and empty_nudges < _MAX_EMPTY_NUDGES):
                empty_nudges += 1
                history.append({"role": "user", "content": _EMPTY_ANSWER_NUDGE})
                trajectory.append(TrajectoryEntry(
                    scenario_id=scenario.scenario_id,
                    step=step_counter,
                    event_type=TrajectoryEventType.ERROR,
                    error=f"empty assistant turn; nudge {empty_nudges}/{_MAX_EMPTY_NUDGES}",
                ))
                continue

            answer_errors = _final_answer_errors(
                action.final_answer or {}, scenario.expected_output_schema, scenario
            )
            if answer_errors:
                if consecutive_protocol_failures < config.max_protocol_retries:
                    protocol_retries += 1
                    consecutive_protocol_failures += 1
                    history.append({
                        "role": "user",
                        "content": _repair_nudge(use_native, answer_errors),
                    })
                    trajectory.append(TrajectoryEntry(
                        scenario_id=scenario.scenario_id,
                        step=step_counter,
                        event_type=TrajectoryEventType.ERROR,
                        error=(
                            "final-answer schema repair "
                            f"{consecutive_protocol_failures}/{config.max_protocol_retries}: "
                            + "; ".join(answer_errors[:6])
                        ),
                    ))
                    continue
                trajectory.append(TrajectoryEntry(
                    scenario_id=scenario.scenario_id,
                    step=step_counter,
                    event_type=TrajectoryEventType.ERROR,
                    error=(
                        "protocol recovery exhausted: final answer failed schema: "
                        + "; ".join(answer_errors[:6])
                    ),
                ))
                # A model-produced but schema-invalid final answer is an
                # incorrect benchmark answer, not infrastructure failure.
                prediction = action.final_answer or {}
                format_failure = True
                state = RolloutState.COMPLETED
                break

            prediction = action.final_answer or {}
            consecutive_protocol_failures = 0
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.FINAL_ANSWER,
                agent_message={
                    "thought": action.thought,
                    "final_answer": prediction,
                },
            ))
            state = RolloutState.COMPLETED
            break

        # ── Tool calls ────────────────────────────────────────────
        if action.has_tool_calls:
            remaining_tool_calls = config.max_tool_calls - total_tool_calls
            if remaining_tool_calls <= 0:
                if use_native and action.tool_calls:
                    for tc in action.tool_calls:
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": "[Tool not executed: total tool-call budget exhausted.]",
                        })
                if consecutive_protocol_failures < config.max_protocol_retries:
                    protocol_retries += 1
                    consecutive_protocol_failures += 1
                    history.append({"role": "user", "content": _TOOL_BUDGET_NUDGE})
                    trajectory.append(TrajectoryEntry(
                        scenario_id=scenario.scenario_id,
                        step=step_counter,
                        event_type=TrajectoryEventType.ERROR,
                        error=(
                            "tool budget exhausted; final-answer repair "
                            f"{consecutive_protocol_failures}/{config.max_protocol_retries}"
                        ),
                    ))
                    continue
                trajectory.append(TrajectoryEntry(
                    scenario_id=scenario.scenario_id,
                    step=step_counter,
                    event_type=TrajectoryEventType.ERROR,
                    error="tool budget exhausted before a valid final answer",
                ))
                format_failure = True
                prediction = {}
                state = RolloutState.COMPLETED
                break

            consecutive_protocol_failures = 0
            state = RolloutState.TOOL_CALLING
            execution_limit = min(
                remaining_tool_calls,
                config.max_tool_calls_per_turn,
            )
            execution_action = AgentAction(
                action_type=ActionType.TOOL_CALL,
                tool_calls=list(action.tool_calls or [])[:execution_limit],
            )
            observations = env.step(execution_action)
            covered_ids: set[str] = set()

            for obs in observations:
                total_tool_calls += 1
                if obs.tool_call_id:
                    covered_ids.add(obs.tool_call_id)
                # Log tool call + observation
                tc = None
                if action.tool_calls:
                    tc_match = [t for t in action.tool_calls if t.tool_call_id == obs.tool_call_id]
                    if tc_match:
                        tc = {
                            "name": tc_match[0].name,
                            "arguments": tc_match[0].arguments,
                        }

                trajectory.append(TrajectoryEntry(
                    scenario_id=scenario.scenario_id,
                    step=step_counter,
                    event_type=TrajectoryEventType.TOOL_CALL,
                    tool_call=tc,
                    observation={
                        "status": obs.status,
                        "content": obs.content,
                        "latency_ms": obs.latency_ms,
                        "error": obs.error,
                    },
                ))

                # Tool result goes into history for the model to see.
                if use_native:
                    # Native protocol: result must be a ``role: "tool"``
                    # message referencing the originating tool_call_id.
                    history.extend(_native_tool_messages(obs))
                else:
                    history.append(obs.to_message())

            # Native protocol safety: the API rejects an assistant turn
            # whose tool_calls are not ALL answered.  Backfill any
            # tool_call that the environment did not execute (e.g. capped
            # by max_tool_calls_per_turn) with a stub tool result.
            if use_native and action.tool_calls:
                for tc in action.tool_calls:
                    if tc.tool_call_id not in covered_ids:
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": "[Tool not executed: call budget reached.]",
                        })

            state = RolloutState.RUNNING
            if total_tool_calls >= config.max_tool_calls:
                history.append({"role": "user", "content": _TOOL_BUDGET_NUDGE})

        # ── Memory update ─────────────────────────────────────────
        elif action.is_memory_update:
            consecutive_protocol_failures = 0
            state = RolloutState.MEMORY_UPDATING
            obs = env.update_memory(action)
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.MEMORY_UPDATE,
                agent_message={"memory_content": action.memory_content},
                observation={"status": obs.status, "content": obs.content},
            ))
            history.append(obs.to_message())
            state = RolloutState.RUNNING

        # ── Invalid action ────────────────────────────────────────
        else:
            if consecutive_protocol_failures < config.max_protocol_retries:
                protocol_retries += 1
                consecutive_protocol_failures += 1
                history.append({
                    "role": "user",
                    "content": _repair_nudge(use_native),
                })
                trajectory.append(TrajectoryEntry(
                    scenario_id=scenario.scenario_id,
                    step=step_counter,
                    event_type=TrajectoryEventType.ERROR,
                    error=(
                        "unparseable action; protocol repair "
                        f"{consecutive_protocol_failures}/{config.max_protocol_retries}"
                    ),
                ))
                continue
            obs = env.invalid_action(action)
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.ERROR,
                error=obs.error,
            ))
            trajectory.append(TrajectoryEntry(
                scenario_id=scenario.scenario_id,
                step=step_counter,
                event_type=TrajectoryEventType.ERROR,
                error="protocol recovery exhausted: no parseable action",
            ))
            format_failure = True
            prediction = {}
            state = RolloutState.COMPLETED
            break

    else:
        trajectory.append(TrajectoryEntry(
            scenario_id=scenario.scenario_id,
            step=step_counter,
            event_type=TrajectoryEventType.ERROR,
            error="max turns exceeded before a valid final answer",
        ))
        # No final answer within the task budget is an empty model answer.
        # The paper specifies that empty/unparseable answers are incorrect and
        # remain in the metric denominator.
        format_failure = True
        prediction = {}
        state = RolloutState.COMPLETED

    wall_time = time.monotonic() - t_start

    # ── Invalid-run gate ──────────────────────────────────────────
    # Only infrastructure/adapter failures are invalid and require resume.
    # Model format failures are completed empty/wrong answers and are scored.
    invalid = state == RolloutState.FAILED
    if invalid:
        prediction = {}
    error_reason: str | None = None
    if invalid or format_failure:
        for te in reversed(trajectory):
            if getattr(te, "error", None):
                error_reason = te.error
                break

    # ── Evaluate (skipped for invalid runs) ───────────────────────
    scores: list[EvalScore] = []
    if not invalid:
        for evaluator in evaluators:
            try:
                score = evaluator.evaluate(scenario, prediction)
                # An evaluator may return None to declare itself "not
                # applicable" for this scenario; skip it so it does not
                # contribute to the metric.
                if score is not None:
                    scores.append(score)
            except Exception as e:
                logger.error(f"[{scenario.scenario_id}] Evaluator {evaluator.metric_name} failed: {e}")
                scores.append(EvalScore(
                    metric=evaluator.metric_name,
                    value=0.0,
                    details={"error": str(e)},
                ))

    usage: dict[str, Any] = {
        "tool_calls": total_tool_calls,
        "turns": step_counter,
        "wall_time_sec": round(wall_time, 2),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "protocol_retries": protocol_retries,
        "format_failure": format_failure,
    }
    if total_reasoning_tokens > 0:
        usage["reasoning_tokens"] = total_reasoning_tokens

    result = EvalResult(
        scenario_id=scenario.scenario_id,
        status=state.value,
        prediction=prediction,
        scores=scores,
        usage=usage,
        error=error_reason,
        invalid=invalid,
    )

    return result, trajectory
