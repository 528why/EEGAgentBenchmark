"""Shared base for the T2–T4 record-level EEG classification tasks.

This base implements ONLY the infrastructure that is identical across
classification sub-tasks:

- The fixed *Rules* / *Action Format* / *Tool Reference* scaffolding.
- The user prompt structure (clinical context + record metadata + question
  + expected output schema, with a metadata whitelist to prevent label
  leakage).
- The ``parse_final_answer`` JSON parsing logic.

Subclasses are expected to provide the **task-specific system instructions**
via :attr:`system_prompt_template` (a string with one ``{tool_overview}``
placeholder) **or** by overriding :meth:`build_system_prompt`.  This keeps
the prompt of each sub-task editable independently — for example, the
analysis guidance for "routine normal/abnormal impression" is *not* the
same as the guidance for "AD/FTD/HC cohort discrimination", and we should
not force them through one shared template.
"""

from __future__ import annotations

import abc
import json
from typing import Any

from eeg_agent_bench.tasks.base import BaseTask
from eeg_agent_bench.tasks.prompt_builder import PromptBuilder, build_tool_overview
from eeg_agent_bench.types import Scenario

# ── Metadata whitelist for user prompt ────────────────────────────────
# Only these keys are rendered to the model.  Keys that leak diagnostic
# labels or classification hints (label_binary, condition, subject_type,
# dataset internal flags) are deliberately excluded to prevent data leakage.
_SAFE_METADATA_KEYS: frozenset[str] = frozenset({
    # Recording parameters
    "channels", "channel_names", "sampling_rate", "n_samples", "duration_sec",
    "montage", "reference",
    # Subject demographics (clinical context, not labels)
    "age", "gender", "sex",
    # NOTE: ``mmse`` is intentionally NOT whitelisted.  The MMSE cognitive
    # score is a near-label shortcut for the T4-Dementia AD/FTD/HC decision and
    # would let a model bypass EEG analysis.  Keep it out of the user prompt.
    # NOTE: ``recording_type`` is intentionally NOT whitelisted.  In Bonn it is
    # perfectly colinear with the label (surface==normal, intracranial==
    # epileptic), so exposing it is a 100% label shortcut.  Keep it out.
})


class ClassificationTask(BaseTask, abc.ABC):
    """Abstract base for record-level classification tasks.

    Concrete subclasses must define:

    - ``task_id`` and ``task_name`` properties (from BaseTask).
    - ``system_prompt_template`` — a string containing ``{tool_overview}``
      placeholder. The template should describe role, rules and analysis
      guidance specific to that sub-task.
    """

    # ── Subclass contract ────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def system_prompt_template(self) -> str:
        """The task-specific system prompt template.

        Must contain ``{tool_overview}`` exactly once.  All other
        formatting placeholders should be escaped (use ``{{`` / ``}}``).
        """
        ...

    # Optional default: subclasses may override to inject extra
    # context (e.g. cohort-specific reminder banners).
    def build_extra_user_sections(self, scenario: Scenario) -> list[str]:
        """Return optional extra sections appended to the user prompt.

        Default returns no extra sections.  Subclasses may override to
        inject task-specific reminders that travel with the user message
        (e.g. a "non-diagnostic" reminder for cohort discrimination).
        """
        return []

    # ── Core: build_messages ─────────────────────────────────────────

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
        """Format the task-specific template with the tool overview."""
        overview = build_tool_overview(tool_specs)
        return self.system_prompt_template.format(tool_overview=overview)

    # ── User prompt assembly (shared) ────────────────────────────────

    @staticmethod
    def _classification_hint(scenario: Scenario) -> str:
        """Return the dataset-specific classification labels."""
        schema = scenario.expected_output_schema or {}
        props = schema.get("properties", {})
        classification = props.get("classification", {})
        labels = classification.get("enum", [])
        if labels:
            return "|".join(str(label) for label in labels)
        return "string"

    def _build_user_prompt(self, scenario: Scenario) -> str:
        parts: list[str] = []

        # ── Clinical context ──────────────────────────────────────────
        clinical_context = scenario.input.get(
            "clinical_context", "No clinical context provided."
        )
        parts.append(f"## Clinical Context\n{clinical_context}")

        # ── EEG record details ────────────────────────────────────────
        if scenario.records:
            parts.append("\n## EEG Record")
            for rec in scenario.records:
                parts.append(f"- **Record ID**: `{rec.record_id}`")
                # NOTE: data_path is intentionally NOT shown — directory
                # names can leak labels (e.g. ".../bonn/S/..." reveals
                # epileptic).
                if rec.metadata:
                    for key, val in rec.metadata.items():
                        if key not in _SAFE_METADATA_KEYS:
                            continue
                        label = key.replace("_", " ").title()
                        parts.append(f"- **{label}**: {val}")
                if rec.role != "target":
                    parts.append(f"- **Role**: {rec.role}")

        # ── Question ──────────────────────────────────────────────────
        question = scenario.input.get(
            "question",
            "Please classify this EEG recording and list supporting findings.",
        )
        parts.append(f"\n## Question\n{question}")

        # ── Optional task-specific reminders ──────────────────────────
        for extra in self.build_extra_user_sections(scenario):
            parts.append(f"\n{extra}")

        # ── Expected output schema ────────────────────────────────────
        classification_hint = self._classification_hint(scenario)
        parts.append(
            "\n## Expected Output Schema\n"
            "```json\n"
            + json.dumps(
                {"classification": classification_hint, "findings": ["string"]},
                ensure_ascii=False,
            )
            + "\n"
            "```"
        )

        return "\n".join(parts)

    # ── Final answer parsing (shared) ────────────────────────────────

    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return {
                "classification": raw.get("classification", ""),
                "findings": raw.get("findings", []),
            }
        if isinstance(raw, str):
            try:
                d = json.loads(raw)
                return self.parse_final_answer(d)
            except (json.JSONDecodeError, TypeError):
                return {"classification": raw.strip(), "findings": []}
        return {"classification": "", "findings": []}


__all__ = ["ClassificationTask"]
