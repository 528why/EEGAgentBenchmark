"""C2-Artifact evaluator — synthetic artifact contamination recognition.

Per-scenario 0/1 correctness (metric ``artifact_correct``); the mean over
scenarios is top-1 accuracy.  Macro-F1, balanced accuracy, confusion and
SNR-stratified breakdowns are computed offline by
:func:`aggregate_artifact_metrics` from the run/score details (the bench's
aggregation layer only averages scalar scores, so the richer 3-class
metrics live here and are consumed by ``scripts``/reporting).

Gold-bearing fields (label / artifact_source / snr_db) are nested under
the reserved ``gold`` key so ``--redact-gold`` strips them from public
exports while workspace ``runs.jsonl`` (un-redacted) keeps them for the
offline 3-class report.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.types import EvalScore, Scenario

LABELS = ("clean", "ocular_contaminated", "muscle_contaminated")
_CONTAM = ("ocular_contaminated", "muscle_contaminated")

# Answerability threshold (fairness).  Contaminated epochs whose SNR is at
# or above this value are "subtle": after the help-level alignment removed
# the feature→label cheat-sheet from the prompt, weak-contamination epochs
# may not be recoverable from the signal alone, so they are reported
# separately and excluded from the *answerable* accuracy.  Clean epochs
# (snr_db is None) are always answerable.  The SNR grid is [-7, -3, 2], so
# the default isolates the +2 dB tier as subtle.
SUBTLE_SNR_MIN_DB: float = 2.0


def is_subtle(gold_label: str, snr_db: Any) -> bool:
    """Whether a (contaminated) epoch is too weak to be reliably answerable."""
    return (
        gold_label in _CONTAM
        and snr_db is not None
        and float(snr_db) >= SUBTLE_SNR_MIN_DB
    )

# Alias map → canonical label.  Matched on lower-cased, whitespace/
# punctuation-normalised tokens.
_ALIASES: dict[str, str] = {
    "clean": "clean",
    "normal": "clean",
    "no_artifact": "clean",
    "none": "clean",
    "ocular_contaminated": "ocular_contaminated",
    "ocular": "ocular_contaminated",
    "eog": "ocular_contaminated",
    "eye": "ocular_contaminated",
    "eye_movement": "ocular_contaminated",
    "eyeblink": "ocular_contaminated",
    "blink": "ocular_contaminated",
    "muscle_contaminated": "muscle_contaminated",
    "muscle": "muscle_contaminated",
    "emg": "muscle_contaminated",
    "myogenic": "muscle_contaminated",
}


def normalise_label(raw: Any) -> str:
    """Return a canonical C2-Artifact label or '' if unrecoverable."""
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    if not text:
        return ""
    token = re.sub(r"[\s\-]+", "_", re.sub(r"[^\w\s\-]", " ", text)).strip("_")
    if token in _ALIASES:
        return _ALIASES[token]
    # canonical label appearing as a substring (e.g. "ocular_contaminated (eog)")
    for canon in LABELS:
        if canon in token:
            return canon
    # any underscore-delimited word matching an alias (e.g. "emg_artifact")
    for part in token.split("_"):
        if part in _ALIASES:
            return _ALIASES[part]
    return ""


class ArtifactClassificationEvaluator(BaseEvaluator):
    """Per-scenario 0/1 correctness for the 3-class artifact decision."""

    @property
    def metric_name(self) -> str:
        return "artifact_correct"

    def evaluate(
        self,
        scenario: Scenario,
        prediction: dict[str, Any],
    ) -> EvalScore:
        gold = scenario.gold_private or {}
        gold_label = normalise_label(gold.get("label", ""))

        raw_pred = None
        if isinstance(prediction, dict):
            raw_pred = (
                prediction.get("classification")
                or prediction.get("label")
                or prediction.get("raw_response")
            )
        else:
            raw_pred = prediction
        pred_label = normalise_label(raw_pred)

        correct = int(pred_label != "" and pred_label == gold_label)

        snr_db = gold.get("snr_db")
        subtle = is_subtle(gold_label, snr_db)
        # Separate the three failure modes (GPT/fairness review): a blank /
        # unparseable prediction is a *format failure*, not a content error;
        # a subtle (weak-SNR) contaminated epoch may be *unanswerable*.
        if pred_label == "":
            status = "format_failure"
        elif correct:
            status = "correct"
        else:
            status = "wrong"

        return EvalScore(
            metric="artifact_correct",
            value=float(correct),
            details={
                # reserved → redacted on --redact-gold
                "gold": {
                    "label": gold_label,
                    "artifact_source": gold.get("artifact_source"),
                    "snr_db": snr_db,
                },
                "predicted": pred_label,
                "predicted_raw": str(raw_pred)[:200] if raw_pred is not None else "",
                "correct": bool(correct),
                "status": status,             # correct | wrong | format_failure
                "answerable": (not subtle),   # False for weak-SNR contaminated
                "subtle": subtle,
            },
        )


# ── Offline 3-class aggregation ──────────────────────────────────────


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def aggregate_artifact_metrics(pairs: list[tuple[str, str, Any]]) -> dict[str, Any]:
    """Compute 3-class diagnostics from (gold, pred, snr_db) triples.

    ``gold`` / ``pred`` are canonical labels (use :func:`normalise_label`
    on raw predictions first).  Empty predictions count as wrong and as a
    distinct ``__none__`` column in the confusion matrix.
    """
    n = len(pairs)
    correct = sum(1 for g, p, _ in pairs if p == g and g)
    accuracy = correct / n if n else 0.0

    # ── Three failure modes separated (fairness) ─────────────────────
    # Restrict to scored items (gold is a known label).
    scored = [(g, p, s) for g, p, s in pairs if g in LABELS]
    n_format_failure = sum(1 for g, p, _ in scored if p == "")
    # Accuracy among items the model actually answered (excludes blanks):
    answered = [(g, p) for g, p, _ in scored if p != ""]
    accuracy_answered = (
        sum(1 for g, p in answered if g == p) / len(answered) if answered else 0.0
    )
    # Answerable subset: drop weak-SNR (subtle) contaminated epochs that may
    # not be recoverable from the signal once the prompt no longer leaks the
    # feature→label mapping.  Blanks still count as wrong within this subset.
    answerable = [(g, p) for g, p, s in scored if not is_subtle(g, s)]
    accuracy_answerable = (
        sum(1 for g, p in answerable if g == p) / len(answerable)
        if answerable else 0.0
    )
    n_subtle = len(scored) - len(answerable)

    # confusion + per-class tp/fp/fn
    conf: dict[str, dict[str, int]] = {g: defaultdict(int) for g in LABELS}
    per_recall_den: dict[str, int] = {g: 0 for g in LABELS}
    per_recall_num: dict[str, int] = {g: 0 for g in LABELS}
    tp = {c: 0 for c in LABELS}
    fp = {c: 0 for c in LABELS}
    fn = {c: 0 for c in LABELS}
    for g, p, _ in pairs:
        if g not in LABELS:
            continue
        col = p if p in LABELS else "__none__"
        conf[g][col] += 1
        per_recall_den[g] += 1
        if p == g:
            per_recall_num[g] += 1
            tp[g] += 1
        else:
            fn[g] += 1
            if p in LABELS:
                fp[p] += 1

    per_class_recall = {
        c: (per_recall_num[c] / per_recall_den[c] if per_recall_den[c] else 0.0)
        for c in LABELS
    }
    per_class_f1 = {c: _f1(tp[c], fp[c], fn[c]) for c in LABELS}
    macro_f1 = sum(per_class_f1.values()) / len(LABELS)
    balanced_accuracy = sum(per_class_recall.values()) / len(LABELS)

    # diagnostic rates
    clean_total = per_recall_den["clean"]
    clean_correct = per_recall_num["clean"]
    contam = [(g, p) for g, p, _ in pairs if g in ("ocular_contaminated", "muscle_contaminated")]
    overcall = sum(1 for g, p, _ in pairs if g == "clean" and p in ("ocular_contaminated", "muscle_contaminated"))
    miss = sum(1 for g, p in contam if p == "clean")
    om_conf = sum(
        1 for g, p in contam
        if p in ("ocular_contaminated", "muscle_contaminated") and p != g
    )

    # SNR stratified accuracy (contaminated only)
    by_snr: dict[Any, dict[str, float]] = {}
    snr_groups: dict[Any, list[tuple[str, str]]] = defaultdict(list)
    for g, p, snr in pairs:
        if snr is not None:
            snr_groups[snr].append((g, p))
    for snr, gp in sorted(snr_groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        c = sum(1 for g, p in gp if g == p)
        by_snr[str(snr)] = {"n": len(gp), "accuracy": c / len(gp) if gp else 0.0}

    return {
        "n": n,
        "accuracy": accuracy,
        # fairness stratification: keep the three failure modes apart
        "n_format_failure": n_format_failure,
        "format_failure_rate": (n_format_failure / len(scored)) if scored else 0.0,
        "accuracy_answered": accuracy_answered,
        "n_subtle": n_subtle,
        "n_answerable": len(answerable),
        "accuracy_answerable": accuracy_answerable,
        "subtle_snr_min_db": SUBTLE_SNR_MIN_DB,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "ocular_recall": per_class_recall["ocular_contaminated"],
        "muscle_recall": per_class_recall["muscle_contaminated"],
        "clean_specificity": (clean_correct / clean_total) if clean_total else 0.0,
        "artifact_overcall_rate": (overcall / clean_total) if clean_total else 0.0,
        "artifact_miss_rate": (miss / len(contam)) if contam else 0.0,
        "ocular_muscle_confusion": (om_conf / len(contam)) if contam else 0.0,
        "confusion": {g: dict(conf[g]) for g in LABELS},
        "score_by_snr": by_snr,
    }


__all__ = [
    "ArtifactClassificationEvaluator",
    "aggregate_artifact_metrics",
    "normalise_label",
    "LABELS",
]
