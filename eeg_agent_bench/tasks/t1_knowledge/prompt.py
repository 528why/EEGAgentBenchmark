"""T1 — EEG Knowledge QA (single-answer 4-option MCQ).

A *knowledge-floor* task: the agent answers EEG / clinical-neurophysiology
multiple-choice questions with **no EEG analysis tools** (ACNS terminology,
waveform/rhythm facts, sleep staging rules, evoked potentials, critical-care
EEG patterns, etc.).  Gold = MedMCQA native correct option (exact match on
the chosen letter).

T1 measures the domain knowledge that the signal-analysis tasks presuppose:
a model that does not know "3 Hz spike-and-wave => absence seizure" cannot be
expected to reach that judgement from signals. T1 therefore helps distinguish
knowledge gaps from agentic-analysis gaps.
"""

from __future__ import annotations

import json
import re
from typing import Any

from eeg_agent_bench.tasks.base import BaseTask
from eeg_agent_bench.tasks.prompt_builder import PromptBuilder
from eeg_agent_bench.types import Scenario

T1_SYSTEM_PROMPT = """\
You are an expert in EEG and clinical neurophysiology taking a \
single-best-answer multiple-choice knowledge exam.

## Rules
1. Each question has exactly FOUR options labelled A, B, C, D, and exactly \
ONE is correct.
2. Choose the single best option using your domain knowledge. No analysis \
tools are available.
3. Think briefly if helpful, but you MUST end your reply with the answer \
JSON block below.

## Answer Format
End your reply with exactly this JSON block (only the option letter):
```json
{"action": "final_answer", "answer": "<A|B|C|D>"}
```
"""

_LETTERS = ("A", "B", "C", "D")


class T1KnowledgeQATask(BaseTask):
    """T1: EEG knowledge multiple-choice QA (no tools)."""

    @property
    def task_id(self) -> str:
        return "T1"

    @property
    def task_name(self) -> str:
        return "eeg_knowledge_qa"

    # ── Messages ──────────────────────────────────────────────────
    def build_messages(
        self,
        scenario: Scenario,
        tool_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # T1 ignores tool_specs (knowledge-only task).
        user_prompt = self._build_user_prompt(scenario)
        return PromptBuilder.build_initial_messages(
            scenario=scenario,
            system_prompt=T1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tool_specs=None,
        )

    @staticmethod
    def _build_user_prompt(scenario: Scenario) -> str:
        question = scenario.input.get("question", "")
        options: dict[str, str] = scenario.input.get("options", {})
        parts = ["## Question", question.strip(), "", "## Options"]
        for letter in _LETTERS:
            if letter in options:
                parts.append(f"{letter}. {options[letter]}")
        parts.append(
            '\nReply with the single correct option letter as JSON: '
            '```json\n{"action": "final_answer", "answer": "<A|B|C|D>"}\n```'
        )
        return "\n".join(parts)

    # ── Final-answer parsing ──────────────────────────────────────
    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            ans = raw.get("answer", raw.get("classification", ""))
            return {"answer": self._coerce_letter(ans)}
        if isinstance(raw, str):
            try:
                return self.parse_final_answer(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                return {"answer": self._coerce_letter(raw)}
        return {"answer": ""}

    @staticmethod
    def _coerce_letter(s: Any) -> str:
        m = re.search(r"[A-Da-d]", str(s))
        return m.group(0).upper() if m else ""


__all__ = ["T1KnowledgeQATask", "T1_SYSTEM_PROMPT"]
