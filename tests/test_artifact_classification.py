"""Tests for the T2-Artifact evaluator (artifact contamination, 3-class)."""

from __future__ import annotations

from eeg_agent_bench.evaluators.artifact_classification import (
    ArtifactClassificationEvaluator,
    aggregate_artifact_metrics,
    normalise_label,
)
from eeg_agent_bench.types import Scenario


def _scenario(label, snr=None, source=None):
    return Scenario(
        scenario_id=f"c2_artifact:t_{label}",
        task_id="T2-Artifact",
        task_name="artifact_contamination",
        dataset="EEGdenoiseNet",
        gold_private={"label": label, "snr_db": snr, "artifact_source": source},
    )


# ── label normalisation / alias tolerance ────────────────────────────

def test_normalise_canonical():
    assert normalise_label("clean") == "clean"
    assert normalise_label("ocular_contaminated") == "ocular_contaminated"
    assert normalise_label("muscle_contaminated") == "muscle_contaminated"


def test_normalise_aliases():
    assert normalise_label("EOG") == "ocular_contaminated"
    assert normalise_label("eye movement") == "ocular_contaminated"
    assert normalise_label("EMG") == "muscle_contaminated"
    assert normalise_label("muscle") == "muscle_contaminated"
    assert normalise_label("Clean.") == "clean"
    assert normalise_label("normal") == "clean"


def test_normalise_messy():
    assert normalise_label("ocular_contaminated (eog)") == "ocular_contaminated"
    assert normalise_label("emg artifact") == "muscle_contaminated"
    assert normalise_label("garbage") == ""
    assert normalise_label("") == ""
    assert normalise_label(None) == ""


# ── per-scenario evaluator ───────────────────────────────────────────

def test_evaluator_correct():
    ev = ArtifactClassificationEvaluator()
    s = _scenario("ocular_contaminated", snr=-3, source="EOG")
    es = ev.evaluate(s, {"classification": "EOG"})
    assert es.value == 1.0
    assert es.details["predicted"] == "ocular_contaminated"


def test_evaluator_wrong():
    ev = ArtifactClassificationEvaluator()
    s = _scenario("muscle_contaminated", snr=2, source="EMG")
    es = ev.evaluate(s, {"classification": "clean"})
    assert es.value == 0.0


def test_evaluator_accepts_label_key():
    ev = ArtifactClassificationEvaluator()
    s = _scenario("clean")
    es = ev.evaluate(s, {"label": "clean"})
    assert es.value == 1.0


def test_evaluator_gold_nested_for_redaction():
    ev = ArtifactClassificationEvaluator()
    s = _scenario("ocular_contaminated", snr=-7, source="EOG")
    es = ev.evaluate(s, {"classification": "ocular_contaminated"})
    # gold-bearing info nested under the reserved "gold" key
    assert es.details["gold"]["label"] == "ocular_contaminated"
    assert es.details["gold"]["snr_db"] == -7
    from eeg_agent_bench.evaluators.base import redact_details
    assert "gold" not in redact_details(es.details)


# ── offline 3-class aggregation ──────────────────────────────────────

def test_aggregate_perfect():
    pairs = [
        ("clean", "clean", None),
        ("ocular_contaminated", "ocular_contaminated", -3),
        ("muscle_contaminated", "muscle_contaminated", 2),
    ]
    agg = aggregate_artifact_metrics(pairs)
    assert agg["accuracy"] == 1.0
    assert agg["macro_f1"] == 1.0
    assert agg["balanced_accuracy"] == 1.0


def test_aggregate_confusion_and_rates():
    pairs = [
        ("clean", "ocular_contaminated", None),   # overcall
        ("clean", "clean", None),
        ("ocular_contaminated", "clean", -7),      # miss
        ("ocular_contaminated", "muscle_contaminated", -3),  # o/m confusion
        ("muscle_contaminated", "muscle_contaminated", 2),
    ]
    agg = aggregate_artifact_metrics(pairs)
    assert agg["clean_specificity"] == 0.5
    assert agg["artifact_overcall_rate"] == 0.5
    assert abs(agg["artifact_miss_rate"] - 1 / 3) < 1e-9
    assert abs(agg["ocular_muscle_confusion"] - 1 / 3) < 1e-9
    assert agg["confusion"]["clean"]["ocular_contaminated"] == 1


def test_aggregate_snr_stratified():
    pairs = [
        ("ocular_contaminated", "ocular_contaminated", -7),
        ("ocular_contaminated", "clean", -7),
        ("muscle_contaminated", "muscle_contaminated", 2),
    ]
    agg = aggregate_artifact_metrics(pairs)
    assert agg["score_by_snr"]["-7"]["n"] == 2
    assert agg["score_by_snr"]["-7"]["accuracy"] == 0.5
    assert agg["score_by_snr"]["2"]["accuracy"] == 1.0


def test_aggregate_empty_prediction_counts_wrong():
    pairs = [("clean", "", None), ("clean", "clean", None)]
    agg = aggregate_artifact_metrics(pairs)
    assert agg["accuracy"] == 0.5
    assert agg["confusion"]["clean"]["__none__"] == 1


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
