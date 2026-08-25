"""Multiple-choice QA accuracy evaluator (T1 — EEG Knowledge QA).

The agent answers a 4-option single-answer MCQ.  Gold is the MedMCQA
native correct-option letter (``cop`` mapped 0->A .. 3->D), stored in
``scenario.gold_private['answer']``.

Scoring is **exact match on the chosen option letter** — objective,
deterministic, with no LLM judge. The per-instance metric is ``mcq_correct``;
official top-1 Accuracy is recomputed by ``reporting/official.py``.

Answer normalisation is defensive: a model may reply ``"B"``, ``"(B)"``,
``"B."``, ``"Option B"`` or even echo the option *text*.  We first try to
recover a bare letter A-D; failing that we match the reply against the
option texts carried in ``scenario.input['options']``.
"""

from __future__ import annotations

import re
from typing import Any

from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.types import EvalScore, Scenario

_LETTERS = ("A", "B", "C", "D")
# Standalone letter only (word-boundaried) so we never pick the "A" inside
# words like "ANSWER" / "Delta".
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().casefold())


def normalise_answer(raw: Any, options: dict[str, str] | None) -> str:
    """Return a canonical option letter (A-D) or '' if unrecoverable.

    Strategy:
      1. If the reply contains an explicit option letter (A-D), use the
         *first* one found.
      2. Otherwise, match the reply text against the option texts.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""

    # 1) explicit letter — but guard against the letter being the first
    #    char of a word like "Delta" only when it's standalone-ish.
    m = re.fullmatch(r"\(?\s*([A-Da-d])\s*\)?\.?", text)
    if m:
        return m.group(1).upper()
    # letter token at start, e.g. "B) beta waves", "B. ...", "Option C"
    m = re.match(r"\s*(?:option\s*)?\(?([A-Da-d])\)?[\.\):\s]", text)
    if m:
        return m.group(1).upper()

    # 2) match against option texts
    if options:
        t = _norm_text(text)
        # exact text match
        for letter in _LETTERS:
            if letter in options and _norm_text(options[letter]) == t:
                return letter
        # containment (model wrapped the option text in a sentence)
        for letter in _LETTERS:
            if letter in options:
                opt = _norm_text(options[letter])
                if opt and (opt in t or t in opt):
                    return letter

    # 3) last resort: a single standalone A-D letter in the string
    m = _LETTER_RE.search(text.upper())
    if m:
        return m.group(1)
    return ""


class MCQAccuracyEvaluator(BaseEvaluator):
    """Per-scenario 0/1 correctness for single-answer 4-option MCQ."""

    @property
    def metric_name(self) -> str:
        return "mcq_correct"

    def evaluate(
        self,
        scenario: Scenario,
        prediction: dict[str, Any],
    ) -> EvalScore:
        options = scenario.input.get("options") if scenario.input else None

        gold_letter = str(scenario.gold_private.get("answer", "")).strip().upper()

        # prediction may carry the letter under "answer" (preferred) or be
        # a bare string / nested final-answer dict.
        raw_pred = None
        if isinstance(prediction, dict):
            raw_pred = (
                prediction.get("answer")
                or prediction.get("classification")
                or prediction.get("raw_response")
            )
        else:
            raw_pred = prediction

        pred_letter = normalise_answer(raw_pred, options)
        correct = int(pred_letter != "" and pred_letter == gold_letter)

        return EvalScore(
            metric="mcq_correct",
            value=float(correct),
            details={
                # "gold" is a reserved key — redacted in leaderboard export.
                "gold": gold_letter,
                "predicted": pred_letter,
                "predicted_raw": str(raw_pred)[:200] if raw_pred is not None else "",
                "correct": bool(correct),
            },
        )


__all__ = ["MCQAccuracyEvaluator", "normalise_answer"]
