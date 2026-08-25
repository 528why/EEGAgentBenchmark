"""T6-Sleep prompt — whole-night sleep staging + architecture summary.

This prompt is intentionally distinct from T3-Epilepsy / T4-Dementia:

- The output is a 30-s-epoch *sequence* (run-length encoded), not a
  single record-level Impression label.
- The action format declares ``hypnogram_rle`` and ``architecture``
  fields in the ``final_answer`` JSON block.
- The system prompt explicitly states the AASM 5-class label set, the
  RLE format, and the architecture-field formulas, so the parser/
  evaluator contract is locked in the prompt itself.

T6-Sleep does NOT subclass ``ClassificationTask``; it inherits directly
from ``BaseTask`` because the output structure is a sequence + a
summary block, not a single classification + findings.
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

T6_SLEEP_SYSTEM_TEMPLATE = """\
You are an expert sleep-EEG analyst agent.  You are given **one
whole-night PSG recording** and must produce two outputs:

1. A 30-second-epoch hypnogram covering the *entire* recording, scored
   in the AASM 5-class system: ``W`` (wake), ``N1``, ``N2``, ``N3``,
   ``REM``.  The hypnogram MUST be encoded as run-length segments — one
   segment per contiguous block of identical stage labels — so that
   long stretches do not blow up the response length.
2. A brief **architecture summary**: REM_pct, N3_pct, sleep_efficiency.

## Rules
1. Each epoch is exactly **30 seconds** long.  ``n_epochs_total`` is \
provided in the user message; your hypnogram MUST cover all epochs from \
``0`` to ``n_epochs_total - 1`` (inclusive) with **no gaps and no \
overlap**.
2. Use AASM 5-class stage labels exactly as written: ``W``, ``N1``, \
``N2``, ``N3``, ``REM``.  Do NOT introduce other labels (no ``S2``, \
``Stage 4``, ``Wake``, etc).
3. Ground every staging decision in signal-level evidence obtained \
from the analysis tools.
4. **Architecture formulas** (use AASM 5-class counts over the FULL \
recording, expressed as fractions in [0, 1] — NOT percentages):
   * ``REM_pct = (# REM epochs) / n_epochs_total``
   * ``N3_pct  = (# N3 epochs) / n_epochs_total``
   * ``sleep_efficiency = (# non-W epochs) / n_epochs_total``
5. You may plan, call tools, and update memory across multiple turns \
before submitting the final answer.

## AASM 5-class scoring criteria (textbook reference, all 5 stages)

Use these as starting hypotheses, not rigid rules.  EEG-only signatures \
are listed first; signals from EOG / EMG channels are noted where they \
disambiguate stages that look similar on the EEG alone.

- **W** (wake): posterior alpha rhythm 8-13 Hz when eyes closed; \
high EMG tone; eye blinks / saccades on EOG.
- **N1** (drowsy transition): alpha attenuation (<50% of epoch); \
low-amplitude mixed-frequency 2-7 Hz background; slow eye movements \
on EOG; **absence** of spindles, K-complexes, and slow-wave activity.  \
N1 is defined largely by exclusion (no single-feature signature).
- **N2**: sleep spindles (11-16 Hz, ≥0.5s bursts) and/or K-complexes; \
otherwise low-amplitude mixed-frequency background.
- **N3** (slow-wave / deep sleep): slow-wave activity 0.5-2 Hz \
covering ≥20% of the epoch at peak-to-peak amplitude >75 µV.
- **REM**: low-amplitude mixed-frequency EEG resembling N1, often \
with prominent fast theta; **rapid eye movements on EOG**; **lowest \
EMG tone of the night** (atonia); occasional sawtooth waves on \
central channels.

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
  "hypnogram_rle": [
    {{"start_epoch": 0,   "end_epoch": 79,  "stage": "W"}},
    {{"start_epoch": 80,  "end_epoch": 91,  "stage": "N1"}},
    {{"start_epoch": 92,  "end_epoch": 250, "stage": "N2"}}
  ],
  "architecture": {{"REM_pct": 0.18, "N3_pct": 0.13, "sleep_efficiency": 0.86}}
}}
```

You may write reasoning in plain text around the JSON block.  Submit \
exactly **one** ``final_answer`` JSON block when you finish.
"""


# ── Metadata whitelist for user prompt (no label leakage) ─────────────
# Only show recording-level parameters; never expose ``epoch_labels_aasm``
# or ``epoch_label_counts_aasm`` (those are gold).
_SAFE_USER_METADATA_KEYS: frozenset[str] = frozenset({
    "sampling_rate", "n_channels", "channels", "eeg_channels",
    "duration_sec", "n_samples", "epoch_sec", "n_epochs_total",
    "age", "sex", "source", "night",
})


# ── final_answer JSON extraction ──────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _try_extract_final_answer_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a ``final_answer`` JSON block from text."""
    # First pass: any ```json ...``` block whose JSON declares
    # ``"action": "final_answer"``.
    for m in _FINAL_ANSWER_RE.finditer(text):
        payload = m.group(1)
        try:
            d = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict) and d.get("action") == "final_answer":
            return d
    # Fallback: try to parse the whole text as JSON if it looks like one.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class T6SleepTask(BaseTask):
    """T6: whole-night sleep staging + architecture summary."""

    @property
    def task_id(self) -> str:
        return "T6"

    @property
    def task_name(self) -> str:
        return "sleep_staging"

    @property
    def system_prompt_template(self) -> str:
        return T6_SLEEP_SYSTEM_TEMPLATE

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
                # data_path intentionally NOT shown.
                if rec.metadata:
                    for key in sorted(rec.metadata.keys()):
                        if key not in _SAFE_USER_METADATA_KEYS:
                            continue
                        val = rec.metadata[key]
                        label = key.replace("_", " ").title()
                        parts.append(f"- **{label}**: {val}")

        question = scenario.input.get(
            "question",
            "Score the entire night in 30-second epochs (AASM 5-class) and "
            "report the architecture summary.",
        )
        parts.append(f"\n## Question\n{question}")

        # Schema reminder (template, not per-scenario enum like
        # T3-Epilepsy; the schema is invariant for sleep staging).
        parts.append(
            "\n## Expected Output Schema\n"
            "```json\n"
            + json.dumps(
                {
                    "hypnogram_rle": [
                        {"start_epoch": "int", "end_epoch": "int",
                         "stage": "W|N1|N2|N3|REM"}
                    ],
                    "architecture": {
                        "REM_pct": "float in [0,1]",
                        "N3_pct": "float in [0,1]",
                        "sleep_efficiency": "float in [0,1]",
                    },
                },
                ensure_ascii=False,
            )
            + "\n```"
        )

        return "\n".join(parts)

    # ── Final answer parsing ─────────────────────────────────────────

    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        """Normalise the agent's final answer into a dict.

        Always returns a dict carrying ``hypnogram_rle`` (list) and
        ``architecture`` (dict).  Missing pieces are filled with empty
        defaults; the evaluator's parser handles further normalisation
        (gap filling, truncation, schema validation).
        """
        d: dict[str, Any] | None = None
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, str):
            d = _try_extract_final_answer_json(raw)
            if d is None:
                # Try plain JSON parse.
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        d = parsed
                except (json.JSONDecodeError, TypeError):
                    d = None

        d = d or {}
        rle = d.get("hypnogram_rle", [])
        if not isinstance(rle, list):
            rle = []
        arch = d.get("architecture", {}) or {}
        if not isinstance(arch, dict):
            arch = {}
        return {
            "hypnogram_rle": rle,
            "architecture": arch,
        }


__all__ = ["T6SleepTask", "T6_SLEEP_SYSTEM_TEMPLATE"]
