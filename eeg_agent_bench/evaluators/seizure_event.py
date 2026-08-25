"""Seizure event-detection evaluators for C1-Seizure.

Two evaluators support long-record search and clinical safety:

- ``SeizureEventF1Evaluator`` (metric ``seizure_event_f1``) — **primary**.
  Event-level precision / recall / F1 using **any-overlap** matching
  (Shoeb 2009): a gold seizure counts as detected if *any* predicted
  interval overlaps it; a predicted interval is a false alarm if it
  overlaps *no* gold seizure.
- ``SeizureFalseAlarmEvaluator`` (metric ``seizure_false_alarms``) —
  secondary, clinical-safety.  Reports the number of false-alarm events
  (and false alarms per hour).  Lower is better.

Both handle the seizure-free case correctly: an empty gold + empty
prediction is a *perfect* answer (the agent correctly found nothing).

Gold source: ``scenario.gold_private['seizure_events']`` — a list of
``[onset_sec, offset_sec]`` pairs (possibly empty).
Prediction: ``prediction['seizure_events']`` — list of
``{onset_sec, offset_sec}`` dicts (possibly empty).
"""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.types import EvalScore, Scenario


def _as_interval(item: Any) -> tuple[float, float] | None:
    """Coerce one event (dict or pair) to a sorted (onset, offset)."""
    onset = offset = None
    if isinstance(item, dict):
        onset = item.get("onset_sec", item.get("onset", item.get("start")))
        offset = item.get("offset_sec", item.get("offset", item.get("end")))
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        onset, offset = item[0], item[1]
    try:
        o = float(onset)
        f = float(offset)
    except (TypeError, ValueError):
        return None
    return (o, f) if o <= f else (f, o)


def _parse_events(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        return []
    out = []
    for it in raw:
        iv = _as_interval(it)
        if iv is not None:
            out.append(iv)
    return out


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if intervals a and b overlap (inclusive touch counts)."""
    return a[0] <= b[1] and b[0] <= a[1]


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching intervals into a disjoint sorted union."""
    if not intervals:
        return []
    out = []
    for o, f in sorted(intervals):
        if out and o <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], f))
        else:
            out.append((o, f))
    return out


def _total_len(intervals: list[tuple[float, float]]) -> float:
    return sum(f - o for o, f in _merge(intervals))


def _intersection_len(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    """Total length of the temporal intersection of two interval sets."""
    ma, mb = _merge(a), _merge(b)
    total = 0.0
    i = j = 0
    while i < len(ma) and j < len(mb):
        lo = max(ma[i][0], mb[j][0])
        hi = min(ma[i][1], mb[j][1])
        if hi > lo:
            total += hi - lo
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def temporal_dice(
    gold: list[tuple[float, float]],
    pred: list[tuple[float, float]],
) -> float:
    """Time-axis binary-mask Dice on the seizure-positive subset.

    ``Dice = 2·|pred ∩ gold| / (|pred| + |gold|)`` where ``|·|`` is the
    total seconds covered by the *union* of a set's intervals.

    Conventions:
    - both empty (silence on a seizure-free record correctly called) → 1.0;
    - gold non-empty, pred empty (full miss) → 0.0;
    - a giant interval [0, T] over a short true seizure → Dice ≈ 0 (the
      anti-cheat property: penalises over-wide predictions).
    """
    len_g = _total_len(gold)
    len_p = _total_len(pred)
    if len_g == 0.0 and len_p == 0.0:
        return 1.0
    denom = len_g + len_p
    if denom == 0.0:
        return 0.0
    return 2.0 * _intersection_len(gold, pred) / denom


def match_events(
    gold: list[tuple[float, float]],
    pred: list[tuple[float, float]],
) -> dict[str, Any]:
    """Any-overlap event matching → P / R / F1 + TP/FP/FN counts.

    - matched_gold: # gold intervals overlapped by ≥1 pred (true positives)
    - matched_pred: # pred intervals overlapping ≥1 gold (true detections)
    - false_alarms: # pred intervals overlapping NO gold
    - missed:       # gold intervals overlapped by NO pred
    """
    n_gold = len(gold)
    n_pred = len(pred)

    # Seizure-free record, correctly called → perfect.
    if n_gold == 0 and n_pred == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp_gold": 0, "matched_pred": 0, "false_alarms": 0, "missed": 0,
            "n_gold": 0, "n_pred": 0,
        }

    gold_hit = [False] * n_gold
    pred_hit = [False] * n_pred
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if _overlaps(p, g):
                pred_hit[i] = True
                gold_hit[j] = True

    matched_gold = sum(gold_hit)
    matched_pred = sum(pred_hit)
    false_alarms = n_pred - matched_pred
    missed = n_gold - matched_gold

    precision = matched_pred / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    recall = matched_gold / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp_gold": matched_gold, "matched_pred": matched_pred,
        "false_alarms": false_alarms, "missed": missed,
        "n_gold": n_gold, "n_pred": n_pred,
    }


class SeizureEventF1Evaluator(BaseEvaluator):
    """Event-level any-overlap F1 for seizure detection."""

    @property
    def metric_name(self) -> str:
        return "seizure_event_f1"

    def evaluate(self, scenario: Scenario, prediction: dict[str, Any]) -> EvalScore:
        gold = _parse_events(scenario.gold_private.get("seizure_events", []))
        pred = _parse_events(prediction.get("seizure_events", []))
        m = match_events(gold, pred)
        return EvalScore(
            metric="seizure_event_f1",
            value=float(m["f1"]),
            details={
                # "gold" reserved key → redacted in leaderboard outputs.
                "gold": [[o, f] for (o, f) in gold],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "n_gold": m["n_gold"],
                "n_pred": m["n_pred"],
                "false_alarms": m["false_alarms"],
                "missed": m["missed"],
            },
        )


class SeizureFalseAlarmEvaluator(BaseEvaluator):
    """Clinical-safety: false-alarm count (and per-hour rate).

    ``value`` = number of false-alarm events (lower is better).
    ``details.false_alarms_per_hour`` normalises by record duration when
    available (``scenario.records[0].metadata['duration_sec']``).
    """

    @property
    def metric_name(self) -> str:
        return "seizure_false_alarms"

    def evaluate(self, scenario: Scenario, prediction: dict[str, Any]) -> EvalScore:
        gold = _parse_events(scenario.gold_private.get("seizure_events", []))
        pred = _parse_events(prediction.get("seizure_events", []))
        m = match_events(gold, pred)
        false_alarms = int(m["false_alarms"])

        duration_sec = None
        if scenario.records:
            duration_sec = scenario.records[0].metadata.get("duration_sec")
        fa_per_hour = None
        if isinstance(duration_sec, (int, float)) and duration_sec > 0:
            fa_per_hour = false_alarms / (duration_sec / 3600.0)

        return EvalScore(
            metric="seizure_false_alarms",
            value=float(false_alarms),
            details={
                "false_alarms": false_alarms,
                "false_alarms_per_hour": fa_per_hour,
                "n_pred": m["n_pred"],
                "n_gold": m["n_gold"],
                "duration_sec": duration_sec,
            },
        )


class SeizureDiceEvaluator(BaseEvaluator):
    """Temporal Dice for seizure detection.

    ``value`` = time-axis binary-mask Dice (see :func:`temporal_dice`).
    Headline metric on the **seizure-positive** subset (detection +
    localisation accuracy); on seizure-free records it degenerates to
    1.0-if-silent / 0.0-if-any-prediction, so the seizure-free subset is
    reported via FA/h (``SeizureFalseAlarmEvaluator``) instead.
    ``details.has_seizure`` lets the aggregation layer split the corpus.
    """

    @property
    def metric_name(self) -> str:
        return "seizure_dice"

    def evaluate(self, scenario: Scenario, prediction: dict[str, Any]) -> EvalScore:
        gold = _parse_events(scenario.gold_private.get("seizure_events", []))
        pred = _parse_events(prediction.get("seizure_events", []))
        dice = temporal_dice(gold, pred)
        return EvalScore(
            metric="seizure_dice",
            value=float(dice),
            details={
                # "gold" reserved key → redacted in leaderboard outputs.
                "gold": [[o, f] for (o, f) in gold],
                "dice": dice,
                "has_seizure": len(gold) > 0,
                "n_gold": len(gold),
                "n_pred": len(pred),
                "gold_sec": _total_len(gold),
                "pred_sec": _total_len(pred),
                "intersection_sec": _intersection_len(gold, pred),
            },
        )


# ── Offline corpus-level aggregation (4 headline metrics) ─────────────
# Two of C1's four headline metrics are corpus-level and cannot be
# per-scenario evaluators (they need the whole confusion matrix over the
# seizure-positive / seizure-free split).  This mirrors
# ``classification.aggregate_classification_metrics`` so C1 reports the
# Headline sensitivity, Dice@seizure, Macro-F1, and false-alarm summaries from
# the standard run/score(_details) outputs.
#
#   - Dice@seizure ↑  : mean temporal Dice over the seizure-*positive* subset
#                       (detection + localisation quality).
#   - Sensitivity  ↑  : record-level recall on the positive subset — the
#                       fraction of seizure records where the agent flagged
#                       ≥1 event (clinical "did we catch it").
#   - Macro-F1     ↑  : record-level binary {seizure, no_seizure} macro-F1
#                       (balances detection against false alarms).
#   - FAR/h        ↓  : mean false-alarms-per-hour over the seizure-*free*
#                       subset (clinical safety; lower is better).


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def seizure_record(
    gold: list[tuple[float, float]],
    pred: list[tuple[float, float]],
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """Per-record pieces feeding :func:`aggregate_seizure_metrics`.

    Built from *raw* gold / predicted events (+ optional record duration),
    so the aggregator can be driven either from live predictions or from
    the ``score_details`` persisted in ``runs.jsonl`` (see
    :func:`seizure_record_from_details`).
    """
    has_seizure = len(gold) > 0
    m = match_events(gold, pred)
    fa = int(m["false_alarms"])
    fa_ph: float | None = None
    if isinstance(duration_sec, (int, float)) and duration_sec > 0:
        fa_ph = fa / (duration_sec / 3600.0)
    return {
        "has_seizure": has_seizure,
        "pred_positive": len(pred) > 0,
        # Dice is defined only on seizure-positive records.
        "dice": temporal_dice(gold, pred) if has_seizure else None,
        # FAR/h is reported on the seizure-free subset.
        "false_alarms_per_hour": fa_ph if not has_seizure else None,
    }


def seizure_record_from_details(score_details: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a :func:`seizure_record` from a run's ``score_details``.

    Reads the ``seizure_dice`` / ``seizure_false_alarms`` blocks written by
    the per-item evaluators — no gold access needed, so it works on redacted
    leaderboard outputs.
    """
    dd = score_details.get("seizure_dice", {}) or {}
    fa = score_details.get("seizure_false_alarms", {}) or {}
    has_seizure = bool(dd.get("has_seizure") or (dd.get("n_gold", 0) or 0) > 0)
    pred_positive = (dd.get("n_pred", 0) or 0) > 0
    return {
        "has_seizure": has_seizure,
        "pred_positive": pred_positive,
        "dice": dd.get("dice") if has_seizure else None,
        "false_alarms_per_hour": (
            fa.get("false_alarms_per_hour") if not has_seizure else None
        ),
    }


def aggregate_seizure_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Corpus-level C1 metrics: Sensitivity + Dice@seizure + Macro-F1 + FAR/h.

    Args:
        records: per-record dicts (invalid runs already excluded by the
            caller) with keys ``has_seizure`` (gold bool), ``pred_positive``
            (bool), ``dice`` (float|None, positive subset) and
            ``false_alarms_per_hour`` (float|None, seizure-free subset) — as
            produced by :func:`seizure_record` /
            :func:`seizure_record_from_details`.

    Returns a dict with the four headline metrics plus the record-level
    confusion (``tp/fn/fp/tn``), ``specificity``, per-class F1, and counts.
    """
    pos = [r for r in records if r.get("has_seizure")]
    neg = [r for r in records if not r.get("has_seizure")]

    # Record-level binary confusion (positive class = "seizure present").
    tp = sum(1 for r in pos if r.get("pred_positive"))
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r.get("pred_positive"))
    tn = len(neg) - fp

    dices = [r["dice"] for r in pos if r.get("dice") is not None]
    dice_at_seizure = sum(dices) / len(dices) if dices else 0.0

    fars = [
        r["false_alarms_per_hour"]
        for r in neg
        if r.get("false_alarms_per_hour") is not None
    ]
    far_per_hour = sum(fars) / len(fars) if fars else 0.0

    sensitivity = tp / len(pos) if pos else 0.0
    specificity = tn / len(neg) if neg else 0.0

    f1_seizure = _f1(tp, fp, fn)          # positive = seizure
    f1_no_seizure = _f1(tn, fn, fp)       # positive = no_seizure
    present = [f for cond, f in ((pos, f1_seizure), (neg, f1_no_seizure)) if cond]
    macro_f1 = sum(present) / len(present) if present else 0.0

    n = len(records)
    return {
        "n": n,
        "n_seizure": len(pos),
        "n_seizure_free": len(neg),
        # ── four headline metrics ──
        "sensitivity": sensitivity,          # ↑ record-level recall (positives)
        "dice_at_seizure": dice_at_seizure,  # ↑ localisation on positives
        "macro_f1": macro_f1,                # ↑ binary seizure/no_seizure
        "far_per_hour": far_per_hour,        # ↓ false alarms/h (seizure-free)
        # ── diagnostics ──
        "specificity": specificity,
        "accuracy": (tp + tn) / n if n else 0.0,
        "per_class_f1": {"seizure": f1_seizure, "no_seizure": f1_no_seizure},
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


__all__ = [
    "SeizureEventF1Evaluator",
    "SeizureFalseAlarmEvaluator",
    "SeizureDiceEvaluator",
    "match_events",
    "temporal_dice",
    "seizure_record",
    "seizure_record_from_details",
    "aggregate_seizure_metrics",
]
