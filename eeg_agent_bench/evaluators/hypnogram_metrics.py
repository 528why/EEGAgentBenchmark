"""C3-Sleep evaluators — independent metric classes.

Sleep staging metrics use **Macro-F1 (primary) + Cohen's Kappa (secondary)**:

- ``hypnogram_macro_f1``:    per-epoch macro-averaged F1 over AASM 5-class.
- ``hypnogram_cohen_kappa``: per-epoch Cohen's kappa over AASM 5-class —
  chance-corrected agreement ``(po - pe) / (1 - pe)``.  Unlike Macro-F1
  it discounts the "lucky" baseline from predicting the dominant W class,
  so it exposes near-random staging that Macro-F1 (≈0.1–0.17 here) hides.
- ``hypnogram_transition_accuracy``: accuracy of stage *transitions*
  between consecutive epochs (fraction of (e[i] → e[i+1]) transitions
  that match gold).  Auxiliary.
- ``hypnogram_architecture_error``: mean absolute error of the three
  architecture statistics (REM_pct, N3_pct, sleep_efficiency).  Per-field
  errors are exposed in ``details``.  Auxiliary.

All metrics share the RLE → per-epoch expander in
``hypnogram_parser.expand_rle_to_epochs`` so the parser corner cases
(gap fill / overlap / out-of-range) are computed once per evaluator.
The cost of running the parser a few times per scenario is negligible
compared to LLM rollout time, and keeps the evaluators independent.
"""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.evaluators.hypnogram_parser import (
    AASM_LABELS,
    DEFAULT_FILL_LABEL,
    expand_rle_to_epochs,
)
from eeg_agent_bench.types import EvalScore, Scenario


# ── small numeric helpers (no sklearn dep) ───────────────────────────


def _per_class_prf(
    pred: list[str], gold: list[str], classes: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(pred, gold) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred, gold) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred, gold) if p != cls and g == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def _macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    if not per_class:
        return 0.0
    return sum(v["f1"] for v in per_class.values()) / len(per_class)


def _cohen_kappa(
    pred: list[str], gold: list[str], classes: tuple[str, ...]
) -> tuple[float, float, float]:
    """Per-epoch Cohen's kappa over the AASM class set.

    ``kappa = (po - pe) / (1 - pe)`` where ``po`` is the observed
    epoch-level agreement (= accuracy) and ``pe`` is the agreement
    expected by chance from the marginal class frequencies.  Returns
    ``(kappa, po, pe)``.  Degenerate ``pe == 1`` case (a single class
    fills both pred and gold): perfect agreement → 1.0, else → 0.0.
    """
    n = min(len(pred), len(gold))
    if n == 0:
        return 0.0, 0.0, 0.0
    po = sum(1 for p, g in zip(pred, gold) if p == g) / n
    pred_counts = {c: 0 for c in classes}
    gold_counts = {c: 0 for c in classes}
    for p, g in zip(pred[:n], gold[:n]):
        if p in pred_counts:
            pred_counts[p] += 1
        if g in gold_counts:
            gold_counts[g] += 1
    pe = sum((pred_counts[c] / n) * (gold_counts[c] / n) for c in classes)
    if abs(1.0 - pe) < 1e-12:
        return (1.0 if po >= 1.0 - 1e-12 else 0.0), po, pe
    return (po - pe) / (1.0 - pe), po, pe


def _transition_accuracy(pred: list[str], gold: list[str]) -> float:
    """Fraction of (gold[i]→gold[i+1]) transitions matched by pred."""
    if len(pred) < 2 or len(gold) < 2:
        return 0.0
    n = min(len(pred), len(gold)) - 1
    if n <= 0:
        return 0.0
    correct = sum(
        1
        for i in range(n)
        if pred[i] == gold[i] and pred[i + 1] == gold[i + 1]
    )
    return correct / n


def _architecture_from_labels(labels: list[str]) -> dict[str, float]:
    n = len(labels)
    if n == 0:
        return {"REM_pct": 0.0, "N3_pct": 0.0, "sleep_efficiency": 0.0}
    rem = sum(1 for l in labels if l == "REM") / n
    n3 = sum(1 for l in labels if l == "N3") / n
    sleep_eff = sum(1 for l in labels if l in ("N1", "N2", "N3", "REM")) / n
    return {
        "REM_pct": rem,
        "N3_pct": n3,
        "sleep_efficiency": sleep_eff,
    }


def _coerce_arch_value(v: Any) -> float | None:
    """Coerce an architecture-field value, accepting either a fraction
    (e.g. ``0.18``) or a percentage written as ``18`` / ``18%`` /
    ``"18%"``.  Returns ``None`` if nothing reasonable can be parsed.
    """
    if v is None:
        return None
    if isinstance(v, bool):  # bool is a subclass of int — disallow
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x / 100.0 if x > 1.0 else x
    if isinstance(v, str):
        s = v.strip().rstrip("%").strip()
        try:
            x = float(s)
        except ValueError:
            return None
        return x / 100.0 if x > 1.0 else x
    return None


# ── shared parse routine (reused by all three evaluators) ────────────


def _parse_for_eval(
    scenario: Scenario, prediction: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return (pred_labels, gold_labels, parse_meta)."""
    gold_epochs: list[str] = list(scenario.gold_private.get("epoch_labels_aasm", []))
    n_epochs_total = int(
        scenario.gold_private.get("n_epochs_total", len(gold_epochs))
    )
    if n_epochs_total <= 0 and len(gold_epochs) > 0:
        n_epochs_total = len(gold_epochs)

    rle = prediction.get("hypnogram_rle", [])
    parsed = expand_rle_to_epochs(
        rle=rle,
        n_epochs_total=n_epochs_total,
        fill_label=DEFAULT_FILL_LABEL,
    )
    pred_labels = parsed.epoch_labels

    # Length safety: gold and pred must match.  If gold is shorter than
    # pred (manifest drift), truncate pred; if longer, pad pred with W.
    if len(gold_epochs) != len(pred_labels):
        target = len(gold_epochs)
        if len(pred_labels) > target:
            pred_labels = pred_labels[:target]
        else:
            pred_labels = pred_labels + [DEFAULT_FILL_LABEL] * (
                target - len(pred_labels)
            )

    return pred_labels, gold_epochs, {
        "coverage": parsed.coverage,
        "n_valid_segments": parsed.n_valid_segments,
        "n_segments_input": parsed.n_segments_input,
        "rle_warnings": parsed.rle_warnings,
        "n_epochs_total": parsed.n_epochs_total,
    }


# ── Evaluator 1: macro-F1 (primary) ──────────────────────────────────


class HypnogramMacroF1Evaluator(BaseEvaluator):
    """Per-epoch macro-F1 over the AASM 5-class label set."""

    @property
    def metric_name(self) -> str:
        return "hypnogram_macro_f1"

    def evaluate(
        self, scenario: Scenario, prediction: dict[str, Any]
    ) -> EvalScore:
        pred, gold, meta = _parse_for_eval(scenario, prediction)
        per_class = _per_class_prf(pred, gold, AASM_LABELS)
        macro = _macro_f1(per_class)
        epoch_acc = (
            sum(1 for p, g in zip(pred, gold) if p == g) / len(gold)
            if gold
            else 0.0
        )
        return EvalScore(
            metric=self.metric_name,
            value=float(macro),
            details={
                "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_class.items()},
                "epoch_accuracy": round(epoch_acc, 4),
                "coverage": round(meta["coverage"], 4),
                "n_valid_segments": meta["n_valid_segments"],
                "n_segments_input": meta["n_segments_input"],
                "rle_warnings": meta["rle_warnings"][:20],  # cap for log size
                "n_epochs_total": meta["n_epochs_total"],
            },
        )


# ── Evaluator 2: Cohen's kappa (secondary) ───────────────────────────


class HypnogramCohenKappaEvaluator(BaseEvaluator):
    """Per-epoch Cohen's kappa over the AASM 5-class label set.

    Secondary headline metric (paired with Macro-F1).  Chance-corrected,
    so it strips the baseline credit a model earns simply by predicting
    the dominant W class — exposing near-random staging.
    """

    @property
    def metric_name(self) -> str:
        return "hypnogram_cohen_kappa"

    def evaluate(
        self, scenario: Scenario, prediction: dict[str, Any]
    ) -> EvalScore:
        pred, gold, meta = _parse_for_eval(scenario, prediction)
        kappa, po, pe = _cohen_kappa(pred, gold, AASM_LABELS)
        return EvalScore(
            metric=self.metric_name,
            value=float(kappa),
            details={
                "observed_agreement": round(po, 4),
                "expected_agreement": round(pe, 4),
                "n_epochs": len(gold),
                "coverage": round(meta["coverage"], 4),
                "rle_warnings": meta["rle_warnings"][:20],
            },
        )


# ── Evaluator 3: transition accuracy (auxiliary) ─────────────────────


class HypnogramTransitionAccuracyEvaluator(BaseEvaluator):
    """Pairwise transition match: (gold[i], gold[i+1]) == (pred[i], pred[i+1])."""

    @property
    def metric_name(self) -> str:
        return "hypnogram_transition_accuracy"

    def evaluate(
        self, scenario: Scenario, prediction: dict[str, Any]
    ) -> EvalScore:
        pred, gold, meta = _parse_for_eval(scenario, prediction)
        ta = _transition_accuracy(pred, gold)
        return EvalScore(
            metric=self.metric_name,
            value=float(ta),
            details={
                "n_transitions": max(len(gold) - 1, 0),
                "coverage": round(meta["coverage"], 4),
                "rle_warnings": meta["rle_warnings"][:20],
            },
        )


# ── Evaluator 4: architecture absolute error (auxiliary) ─────────────


class HypnogramArchitectureErrorEvaluator(BaseEvaluator):
    """Mean absolute error across REM_pct, N3_pct, sleep_efficiency.

    The metric value is the **negative** mean absolute error so that
    "higher = better" matches the convention used by the reporting
    layer; per-field absolute errors and the raw (predicted, gold)
    values are surfaced in ``details``.

    Two prediction-supply paths are honoured:
    1. ``prediction['architecture']`` block, if the agent reported it.
    2. Otherwise, derive architecture stats from the parsed hypnogram
       (so that even an architecture-less RLE still produces a score).
    """

    @property
    def metric_name(self) -> str:
        return "hypnogram_architecture_error"

    def evaluate(
        self, scenario: Scenario, prediction: dict[str, Any]
    ) -> EvalScore:
        pred_labels, gold_labels, meta = _parse_for_eval(scenario, prediction)

        gold_arch_raw = scenario.gold_private.get("architecture") or {}
        # Fallback: compute from gold labels if missing from gold_private.
        gold_arch = {
            "REM_pct": float(gold_arch_raw.get("REM_pct", 0.0)),
            "N3_pct": float(gold_arch_raw.get("N3_pct", 0.0)),
            "sleep_efficiency": float(gold_arch_raw.get("sleep_efficiency", 0.0)),
        }
        if not gold_arch_raw:
            gold_arch = _architecture_from_labels(gold_labels)

        # Predicted architecture: prefer the block submitted by the
        # agent; coerce noisy inputs (e.g. percentages, strings) and
        # fill missing fields from the parsed hypnogram.
        derived = _architecture_from_labels(pred_labels)
        agent_arch = prediction.get("architecture") or {}
        pred_arch: dict[str, float] = {}
        for field_name in ("REM_pct", "N3_pct", "sleep_efficiency"):
            v = _coerce_arch_value(agent_arch.get(field_name))
            pred_arch[field_name] = float(v) if v is not None else float(derived[field_name])

        per_field_err = {
            k: abs(pred_arch[k] - gold_arch[k]) for k in pred_arch
        }
        mae = sum(per_field_err.values()) / len(per_field_err)

        return EvalScore(
            metric=self.metric_name,
            value=float(-mae),  # higher = better
            details={
                "mae": round(mae, 6),
                "per_field_abs_error": {k: round(v, 6) for k, v in per_field_err.items()},
                "predicted": {k: round(v, 6) for k, v in pred_arch.items()},
                # ``gold`` is a reserved key — stripped by ``redact_details``
                # for leaderboard exports.
                "gold": {k: round(v, 6) for k, v in gold_arch.items()},
                "coverage": round(meta["coverage"], 4),
                "rle_warnings": meta["rle_warnings"][:20],
            },
        )


__all__ = [
    "HypnogramMacroF1Evaluator",
    "HypnogramCohenKappaEvaluator",
    "HypnogramTransitionAccuracyEvaluator",
    "HypnogramArchitectureErrorEvaluator",
]
