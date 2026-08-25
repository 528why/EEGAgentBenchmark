"""T5-Seizure prompt — long-record electrographic seizure detection.

Distinct from the T2–T4 classification tasks and from T6:

- Output is an **open-ended event list** (variable number of seizure
  intervals, possibly empty), not a single label or a fixed-length
  sequence.
- The system prompt frames the task as a **long-record search**: the
  record is ~1 hour at 256 Hz with 23 channels, so the agent must sweep
  it in chunks, localise suspicious segments, then refine onset/offset.
- Clinical-safety framing: over-calling (false alarms) is penalised by
  the secondary metric, so the agent must weigh sensitivity vs
  specificity rather than flagging everything.

T5-Seizure inherits directly from ``BaseTask`` (like T6-Sleep), because
its output structure is an event list, not classification + findings.
"""

from __future__ import annotations

import json
import re
from typing import Any

from eeg_agent_bench.tasks.base import BaseTask
from eeg_agent_bench.tasks.prompt_builder import PromptBuilder, build_tool_overview
from eeg_agent_bench.types import Scenario

# ── System prompt template ────────────────────────────────────────────
# Single ``{tool_overview}`` placeholder; all other braces escaped.

T5_SEIZURE_SYSTEM_TEMPLATE = """\
You are an expert epileptologist agent.  You are given **one long
scalp-EEG recording** (continuous, typically up to ~1 hour, 256 Hz,
~23 bipolar channels) and must report **all electrographic seizures**
in it as time intervals.

## What is an electrographic seizure (EEG signatures)
A seizure is an **evolving, rhythmic, abnormal discharge** that:
- starts with rhythmic activity (often theta/alpha or rhythmic spiking),
- **evolves** in frequency, amplitude, and spatial spread over seconds,
- is **sustained** (usually ≥10 s) and clearly stands out from the
  background,
- often shows post-ictal suppression/slowing afterwards.
Isolated spikes, sharp waves, single artifacts, or brief bursts are
**NOT** seizures — do not report them.

## Rules
1. Report seizures as a list of ``{{"onset_sec": <s>, "offset_sec": <s>}}`` \
intervals, in seconds from the start of the recording.
2. If there is **no** seizure, return an **empty list** ``[]``.  Many \
records are seizure-free; an empty list is a valid, expected answer.
3. Ground every decision in **signal-level evidence** from the tools.
4. **Clinical safety**: avoid over-calling.  Only report a segment as a \
seizure when the evidence shows clear rhythmic *evolution*, not just a \
single transient or noisy stretch.
5. You may plan, call tools, and update memory across multiple turns \
before submitting the final answer.

{tool_overview}

## Data Access
Pass ``record_id`` to tools; tools handle data loading internally.

## Action Format
When you want to call a tool, include this JSON block:
```json
{{"action": "tool_call", "tool_name": "<name>", "arguments": {{...}}}}
```
You will receive the tool result, then continue your analysis.

When you are ready to give your final answer, include this JSON block:
```json
{{
  "action": "final_answer",
  "seizure_events": [
    {{"onset_sec": <onset_in_seconds>, "offset_sec": <offset_in_seconds>}}
  ]
}}
```
For a seizure-free record, submit:
```json
{{"action": "final_answer", "seizure_events": []}}
```

You may write reasoning in plain text around the JSON block.  Submit \
exactly **one** ``final_answer`` JSON block when you finish.
"""


# ── Metadata whitelist for user prompt (no label leakage) ─────────────
# NEVER expose ``seizure_events`` / ``n_seizures`` (gold).
_SAFE_USER_METADATA_KEYS: frozenset[str] = frozenset({
    "sampling_rate", "n_channels", "channels", "channel_names",
    "duration_sec", "montage", "recording_type",
})


# ── final_answer JSON extraction ──────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _try_extract_final_answer_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a ``final_answer`` JSON block from text."""
    for m in _FINAL_ANSWER_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and d.get("action") == "final_answer":
            return d
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _normalise_events(raw_events: Any) -> list[dict[str, float]]:
    """Coerce an arbitrary events payload into a clean interval list.

    Tolerant of: list of dicts ({onset_sec, offset_sec} or {start,end}),
    list of [onset, offset] pairs.  Drops malformed / non-numeric / and
    reversed-or-zero-length intervals are kept as-is but ordered so
    onset ≤ offset.
    """
    events: list[dict[str, float]] = []
    if not isinstance(raw_events, list):
        return events
    for item in raw_events:
        onset = offset = None
        if isinstance(item, dict):
            onset = item.get("onset_sec", item.get("onset", item.get("start_sec", item.get("start"))))
            offset = item.get("offset_sec", item.get("offset", item.get("end_sec", item.get("end"))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            onset, offset = item[0], item[1]
        try:
            o = float(onset)
            f = float(offset)
        except (TypeError, ValueError):
            continue
        if f < o:
            o, f = f, o
        events.append({"onset_sec": o, "offset_sec": f})
    return events


class T5SeizureTask(BaseTask):
    """T5: long-record electrographic seizure event detection."""

    @property
    def task_id(self) -> str:
        return "T5"

    @property
    def task_name(self) -> str:
        return "seizure_event_detection"

    @property
    def system_prompt_template(self) -> str:
        return T5_SEIZURE_SYSTEM_TEMPLATE

    # ── Initial prompt assembly ──────────────────────────────────────

    def build_messages(
        self,
        scenario: Scenario,
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system_prompt = self.build_system_prompt(tool_specs)
        user_prompt = self._build_user_prompt(scenario)
        return PromptBuilder.build_initial_messages(
            scenario=scenario,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tool_specs=tool_specs,
        )

    def build_system_prompt(self, tool_specs: list[dict[str, Any]]) -> str:
        overview = build_tool_overview(tool_specs)
        return self.system_prompt_template.format(tool_overview=overview)

    def _build_user_prompt(self, scenario: Scenario) -> str:
        parts: list[str] = []

        clinical_context = scenario.input.get(
            "clinical_context", "No clinical context provided."
        )
        parts.append(f"## Clinical Context\n{clinical_context}")

        if scenario.records:
            parts.append("\n## EEG Record")
            for rec in scenario.records:
                parts.append(f"- **Record ID**: `{rec.record_id}`")
                if rec.metadata:
                    for key in sorted(rec.metadata.keys()):
                        if key not in _SAFE_USER_METADATA_KEYS:
                            continue
                        val = rec.metadata[key]
                        label = key.replace("_", " ").title()
                        parts.append(f"- **{label}**: {val}")

        question = scenario.input.get(
            "question",
            "Identify all electrographic seizures in this recording and "
            "report each as an onset/offset interval in seconds. If there "
            "is no seizure, return an empty list.",
        )
        parts.append(f"\n## Question\n{question}")

        parts.append(
            "\n## Expected Output Schema\n"
            "```json\n"
            + json.dumps(
                {
                    "seizure_events": [
                        {"onset_sec": "float", "offset_sec": "float"}
                    ]
                },
                ensure_ascii=False,
            )
            + "\n```"
        )

        return "\n".join(parts)

    # ── Final answer parsing ─────────────────────────────────────────

    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        d: dict[str, Any] | None = None
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, str):
            d = _try_extract_final_answer_json(raw)
            if d is None:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        d = parsed
                except (json.JSONDecodeError, TypeError):
                    d = None
        d = d or {}
        return {"seizure_events": _normalise_events(d.get("seizure_events", []))}


__all__ = ["T5SeizureTask", "T5_SEIZURE_SYSTEM_TEMPLATE"]
