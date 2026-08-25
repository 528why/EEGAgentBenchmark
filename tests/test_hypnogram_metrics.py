"""Unit tests for the T6-Sleep hypnogram parser + evaluators.

Covers the sleep-staging parser and metric edge cases:

- E1: all-W prediction vs mixed gold → low macro_f1, W recall=1, others=0.
- E2: perfect prediction vs all-5-class gold → macro_f1=1, ta=1, mae=0.
- E3: swap N2/N3 → macro_f1 drops but >0 (W/REM still correct).
- E4: architecture formulas match v3.1 §四 (verified inside E2).
- E5: parser corner cases (empty / gap / overshoot / overlap / aliases /
      epoch jump) all produce a sensible per-epoch sequence without
      raising.

Run with ``pytest tests/test_hypnogram_metrics.py -v``.
"""

from __future__ import annotations

from eeg_agent_bench.evaluators.hypnogram_metrics import (
    HypnogramArchitectureErrorEvaluator,
    HypnogramCohenKappaEvaluator,
    HypnogramMacroF1Evaluator,
    HypnogramTransitionAccuracyEvaluator,
)
from eeg_agent_bench.evaluators.hypnogram_parser import expand_rle_to_epochs
from eeg_agent_bench.types import EvaluatorConfig, Scenario


def _make_scenario(epoch_labels: list[str], arch: dict[str, float] | None = None) -> Scenario:
    n = len(epoch_labels)
    if arch is None:
        rem = sum(1 for x in epoch_labels if x == "REM") / max(n, 1)
        n3 = sum(1 for x in epoch_labels if x == "N3") / max(n, 1)
        eff = sum(1 for x in epoch_labels if x in ("N1", "N2", "N3", "REM")) / max(n, 1)
        arch = {"REM_pct": rem, "N3_pct": n3, "sleep_efficiency": eff}
    return Scenario(
        scenario_id="test",
        task_id="T6-Sleep",
        task_name="sleep_staging",
        dataset="Sleep-EDFx",
        gold_private={
            "epoch_labels_aasm": list(epoch_labels),
            "n_epochs_total": n,
            "architecture": arch,
        },
        evaluator=EvaluatorConfig(primary="hypnogram_macro_f1"),
    )


# ── E5: parser corner cases ──────────────────────────────────────────


def test_e5_empty_rle_fills_with_W():
    res = expand_rle_to_epochs([], n_epochs_total=10)
    assert res.epoch_labels == ["W"] * 10
    assert res.coverage == 0.0
    assert res.n_valid_segments == 0


def test_e5_gap_filled_with_W():
    res = expand_rle_to_epochs(
        [
            {"start_epoch": 0, "end_epoch": 4, "stage": "N2"},
            {"start_epoch": 7, "end_epoch": 9, "stage": "REM"},
        ],
        n_epochs_total=10,
    )
    assert res.epoch_labels == ["N2"] * 5 + ["W"] * 2 + ["REM"] * 3


def test_e5_overshoot_truncated():
    res = expand_rle_to_epochs(
        [{"start_epoch": 0, "end_epoch": 100, "stage": "W"}],
        n_epochs_total=5,
    )
    assert res.epoch_labels == ["W"] * 5
    assert any("truncated" in w for w in res.rle_warnings)


def test_e5_overlap_later_wins():
    res = expand_rle_to_epochs(
        [
            {"start_epoch": 0, "end_epoch": 9, "stage": "W"},
            {"start_epoch": 3, "end_epoch": 6, "stage": "N3"},
        ],
        n_epochs_total=10,
    )
    assert res.epoch_labels == ["W", "W", "W", "N3", "N3", "N3", "N3", "W", "W", "W"]


def test_e5_bad_stage_skipped():
    res = expand_rle_to_epochs(
        [
            {"start_epoch": 0, "end_epoch": 2, "stage": "XYZ"},
            {"start_epoch": 3, "end_epoch": 5, "stage": "N1"},
        ],
        n_epochs_total=6,
    )
    assert res.epoch_labels == ["W", "W", "W", "N1", "N1", "N1"]


def test_e5_stage_aliases():
    res = expand_rle_to_epochs(
        [
            {"start_epoch": 0, "end_epoch": 1, "stage": "Stage 4"},  # → N3
            {"start_epoch": 2, "end_epoch": 3, "stage": "s2"},        # → N2
        ],
        n_epochs_total=4,
    )
    assert res.epoch_labels == ["N3", "N3", "N2", "N2"]


def test_e5_epoch_jump_filled():
    res = expand_rle_to_epochs(
        [
            {"start_epoch": 0, "end_epoch": 99, "stage": "W"},
            {"start_epoch": 200, "end_epoch": 299, "stage": "N2"},
        ],
        n_epochs_total=300,
    )
    assert res.epoch_labels[:100] == ["W"] * 100
    assert res.epoch_labels[100:200] == ["W"] * 100   # gap → W
    assert res.epoch_labels[200:300] == ["N2"] * 100


# ── E1: all-W prediction ─────────────────────────────────────────────


def test_e1_all_w_prediction():
    sc = _make_scenario(["W"] * 30 + ["N2"] * 40 + ["N3"] * 20 + ["REM"] * 10)
    pred = {
        "hypnogram_rle": [{"start_epoch": 0, "end_epoch": 99, "stage": "W"}],
        "architecture": {"REM_pct": 0.0, "N3_pct": 0.0, "sleep_efficiency": 0.0},
    }
    score = HypnogramMacroF1Evaluator().evaluate(sc, pred)
    assert score.value < 1.0
    assert score.details["per_class"]["W"]["recall"] == 1.0
    assert score.details["per_class"]["N2"]["recall"] == 0.0


# ── E2: perfect prediction (all 5 classes present) ───────────────────


def test_e2_perfect_prediction():
    sc = _make_scenario(
        ["W"] * 20 + ["N1"] * 10 + ["N2"] * 40 + ["N3"] * 20 + ["REM"] * 10,
        arch={"REM_pct": 0.10, "N3_pct": 0.20, "sleep_efficiency": 0.80},
    )
    pred = {
        "hypnogram_rle": [
            {"start_epoch": 0, "end_epoch": 19, "stage": "W"},
            {"start_epoch": 20, "end_epoch": 29, "stage": "N1"},
            {"start_epoch": 30, "end_epoch": 69, "stage": "N2"},
            {"start_epoch": 70, "end_epoch": 89, "stage": "N3"},
            {"start_epoch": 90, "end_epoch": 99, "stage": "REM"},
        ],
        "architecture": {"REM_pct": 0.10, "N3_pct": 0.20, "sleep_efficiency": 0.80},
    }
    mf = HypnogramMacroF1Evaluator().evaluate(sc, pred)
    ta = HypnogramTransitionAccuracyEvaluator().evaluate(sc, pred)
    ae = HypnogramArchitectureErrorEvaluator().evaluate(sc, pred)
    assert abs(mf.value - 1.0) < 1e-6
    assert abs(ta.value - 1.0) < 1e-6
    assert ae.details["mae"] < 1e-6


# ── E3: N2/N3 swapped ────────────────────────────────────────────────


def test_e3_swap_n2_n3():
    sc = _make_scenario(
        ["W"] * 30 + ["N2"] * 40 + ["N3"] * 20 + ["REM"] * 10,
        arch={"REM_pct": 0.10, "N3_pct": 0.20, "sleep_efficiency": 0.70},
    )
    pred = {
        "hypnogram_rle": [
            {"start_epoch": 0, "end_epoch": 29, "stage": "W"},
            {"start_epoch": 30, "end_epoch": 69, "stage": "N3"},  # swapped
            {"start_epoch": 70, "end_epoch": 89, "stage": "N2"},  # swapped
            {"start_epoch": 90, "end_epoch": 99, "stage": "REM"},
        ],
        "architecture": {},
    }
    mf = HypnogramMacroF1Evaluator().evaluate(sc, pred)
    assert 0.0 < mf.value < 1.0
    assert mf.details["per_class"]["W"]["f1"] == 1.0
    assert mf.details["per_class"]["REM"]["f1"] == 1.0


# ── E4: architecture formulas (validated implicitly by E2) ───────────


def test_e4_architecture_formulas_via_perfect_prediction():
    # Mirrors E2 but verifies arch values explicitly.
    sc = _make_scenario(
        ["W"] * 50 + ["N2"] * 30 + ["N3"] * 10 + ["REM"] * 10,
        arch={"REM_pct": 0.10, "N3_pct": 0.10, "sleep_efficiency": 0.50},
    )
    pred = {
        "hypnogram_rle": [
            {"start_epoch": 0, "end_epoch": 49, "stage": "W"},
            {"start_epoch": 50, "end_epoch": 79, "stage": "N2"},
            {"start_epoch": 80, "end_epoch": 89, "stage": "N3"},
            {"start_epoch": 90, "end_epoch": 99, "stage": "REM"},
        ],
        "architecture": {"REM_pct": 0.10, "N3_pct": 0.10, "sleep_efficiency": 0.50},
    }
    ae = HypnogramArchitectureErrorEvaluator().evaluate(sc, pred)
    assert ae.details["per_field_abs_error"]["REM_pct"] < 1e-6
    assert ae.details["per_field_abs_error"]["N3_pct"] < 1e-6
    assert ae.details["per_field_abs_error"]["sleep_efficiency"] < 1e-6


# ── Cohen's kappa (secondary headline metric) ────────────────────────


def test_kappa_perfect_prediction_is_one():
    sc = _make_scenario(["W"] * 50 + ["N2"] * 30 + ["N3"] * 10 + ["REM"] * 10)
    pred = {
        "hypnogram_rle": [
            {"start_epoch": 0, "end_epoch": 49, "stage": "W"},
            {"start_epoch": 50, "end_epoch": 79, "stage": "N2"},
            {"start_epoch": 80, "end_epoch": 89, "stage": "N3"},
            {"start_epoch": 90, "end_epoch": 99, "stage": "REM"},
        ],
    }
    k = HypnogramCohenKappaEvaluator().evaluate(sc, pred)
    assert abs(k.value - 1.0) < 1e-9
    assert abs(k.details["observed_agreement"] - 1.0) < 1e-9


def test_kappa_all_W_on_W_dominant_gold_is_zero():
    # 90% W gold; predicting all-W gets 0.90 accuracy but kappa == 0
    # (no skill beyond the base rate) — this is exactly why kappa is the
    # secondary metric: it strips the W-baseline credit Macro-F1/acc give.
    sc = _make_scenario(["W"] * 90 + ["N2"] * 10)
    pred = {"hypnogram_rle": [{"start_epoch": 0, "end_epoch": 99, "stage": "W"}]}
    k = HypnogramCohenKappaEvaluator().evaluate(sc, pred)
    assert abs(k.details["observed_agreement"] - 0.90) < 1e-9
    assert abs(k.value) < 1e-9


def test_kappa_below_chance_is_negative():
    # systematically wrong → kappa < 0
    sc = _make_scenario(["W"] * 50 + ["N2"] * 50)
    pred = {
        "hypnogram_rle": [
            {"start_epoch": 0, "end_epoch": 49, "stage": "N2"},
            {"start_epoch": 50, "end_epoch": 99, "stage": "W"},
        ]
    }
    k = HypnogramCohenKappaEvaluator().evaluate(sc, pred)
    assert k.value < 0.0
