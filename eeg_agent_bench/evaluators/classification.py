"""Generic record-level classification accuracy evaluator.

Used by Bonn screening and ds004504 cohort classification.
Compares ``prediction.classification`` against
``scenario.gold_private['label']`` after lower-casing and stripping.

The per-instance metric is ``classification_correct``. Official task-level
Accuracy and Macro-F1 are recomputed by ``reporting/official.py``.
"""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.types import EvalScore, Scenario


class ClassificationAccuracyEvaluator(BaseEvaluator):
    """Per-scenario binary correctness for record-level classification.

    Aggregating over scenarios yields top-1 accuracy.  Macro-F1 and
    per-class recall are computed downstream by the reporting layer.
    """

    @property
    def metric_name(self) -> str:
        return "classification_correct"

    def evaluate(
        self,
        scenario: Scenario,
        prediction: dict[str, Any],
    ) -> EvalScore:
        gold_label = str(scenario.gold_private.get("label", "")).strip().lower()
        pred_label = str(prediction.get("classification", "")).strip().lower()

        correct = int(pred_label == gold_label)

        return EvalScore(
            metric="classification_correct",
            value=float(correct),
            details={
                # "gold" is a reserved key — stripped by ``redact_details``
                # when writing leaderboard outputs.
                "gold": gold_label,
                "predicted": pred_label,
                "correct": bool(correct),
            },
        )


# ── Offline corpus-level aggregation (macro-F1) ──────────────────────
# macro-F1 is a corpus-level metric and cannot be a per-scenario
# evaluator (it needs the full confusion matrix).  This mirrors
# ``artifact_classification.aggregate_artifact_metrics`` so C3-Routine
# (normal/epileptic) and cohort (AD/FTD/HC) tasks report the
# headline ``acc + macro-F1`` from the standard run/score outputs.


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def aggregate_classification_metrics(
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compute acc + macro-F1 + per-class F1/recall + confusion matrix.

    Args:
        pairs: list of ``(gold_label, pred_label)`` over the complete task.
            Empty and non-reference predictions must remain in this list and
            are counted as errors. Labels are lower-cased / stripped here.

    Returns a dict with ``accuracy``, ``macro_f1``, ``balanced_accuracy``,
    ``per_class_f1``, ``per_class_recall``, ``confusion``, ``n``, ``labels``.
    """
    norm = [(str(g).strip().lower(), str(p).strip().lower()) for g, p in pairs]
    # The paper defines Macro-F1 over the task's reference label set. Never
    # extend the averaged class set with arbitrary model output.
    labels = sorted({g for g, _ in norm})
    invalid_label = "__invalid__"
    confusion_columns = [*labels, invalid_label]
    n = len(norm)
    correct = sum(1 for g, p in norm if g == p)

    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}
    rec_num = {c: 0 for c in labels}
    rec_den = {c: 0 for c in labels}
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in confusion_columns} for g in labels
    }
    gold_labels = labels
    invalid_predictions = 0
    for g, p in norm:
        rec_den[g] += 1
        if g == p:
            tp[g] += 1
            rec_num[g] += 1
        else:
            fn[g] += 1
            if p in fp:
                fp[p] += 1
        column = p if p in labels else invalid_label
        confusion[g][column] += 1
        invalid_predictions += int(column == invalid_label)

    per_class_f1 = {c: _f1(tp[c], fp[c], fn[c]) for c in gold_labels}
    per_class_recall = {
        c: (rec_num[c] / rec_den[c] if rec_den[c] else 0.0) for c in gold_labels
    }
    macro_f1 = (
        sum(per_class_f1.values()) / len(gold_labels) if gold_labels else 0.0
    )
    balanced_acc = (
        sum(per_class_recall.values()) / len(gold_labels) if gold_labels else 0.0
    )
    return {
        "n": n,
        "labels": gold_labels,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "invalid_predictions": invalid_predictions,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


__all__ = [
    "ClassificationAccuracyEvaluator",
    "aggregate_classification_metrics",
]
