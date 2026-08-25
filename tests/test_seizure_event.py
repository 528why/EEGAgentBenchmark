"""Unit tests for the T5-Seizure event evaluators (pure-python, no deps)."""

from __future__ import annotations

from eeg_agent_bench.evaluators.seizure_event import (
    SeizureEventF1Evaluator,
    SeizureFalseAlarmEvaluator,
    aggregate_seizure_metrics,
    match_events,
    seizure_record,
    seizure_record_from_details,
)
from eeg_agent_bench.types import RecordRef, Scenario


def _scenario(gold_events, duration_sec=3600.0):
    return Scenario(
        scenario_id="t",
        task_id="T5-Seizure",
        task_name="seizure_event_detection",
        dataset="CHB-MIT",
        records=[RecordRef(record_id="EEG_0001", data_path="x.edf",
                           metadata={"duration_sec": duration_sec})],
        gold_private={"seizure_events": gold_events,
                      "has_seizure": bool(gold_events)},
    )


# ── match_events core ────────────────────────────────────────────────

def test_perfect_single_hit():
    m = match_events([(100, 140)], [(95, 150)])
    assert m["f1"] == 1.0
    assert m["false_alarms"] == 0
    assert m["missed"] == 0


def test_seizure_free_correct():
    # empty gold + empty pred = perfect
    m = match_events([], [])
    assert m["f1"] == 1.0
    assert m["false_alarms"] == 0


def test_seizure_free_false_alarm():
    # no gold but agent reported one → all false alarm, f1=0
    m = match_events([], [(10, 20)])
    assert m["f1"] == 0.0
    assert m["false_alarms"] == 1
    assert m["precision"] == 0.0


def test_missed_seizure():
    # gold present, pred empty → recall 0, f1 0
    m = match_events([(100, 140)], [])
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["missed"] == 1


def test_partial_overlap_counts_as_hit():
    # any-overlap: a 1-second overlap still detects the gold
    m = match_events([(100, 200)], [(199, 400)])
    assert m["tp_gold"] == 1
    assert m["recall"] == 1.0


def test_no_overlap_is_false_alarm():
    m = match_events([(100, 140)], [(500, 540)])
    assert m["missed"] == 1
    assert m["false_alarms"] == 1
    assert m["f1"] == 0.0


def test_two_preds_one_gold_one_fa():
    # one pred hits the gold, the other is a false alarm
    m = match_events([(100, 140)], [(110, 130), (800, 820)])
    assert m["tp_gold"] == 1
    assert m["matched_pred"] == 1
    assert m["false_alarms"] == 1
    # precision = 1/2, recall = 1 → f1 = 2/3
    assert abs(m["f1"] - (2 * 0.5 * 1.0 / 1.5)) < 1e-9


def test_two_gold_one_pred_covers_both():
    # one wide pred overlaps two gold events
    m = match_events([(100, 140), (200, 240)], [(90, 260)])
    assert m["tp_gold"] == 2
    assert m["matched_pred"] == 1
    assert m["false_alarms"] == 0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0


# ── Evaluator wrappers ───────────────────────────────────────────────

def test_f1_evaluator_perfect():
    ev = SeizureEventF1Evaluator()
    sc = _scenario([[100, 140]])
    s = ev.evaluate(sc, {"seizure_events": [{"onset_sec": 95, "offset_sec": 150}]})
    assert s.metric == "seizure_event_f1"
    assert s.value == 1.0


def test_f1_evaluator_seizure_free():
    ev = SeizureEventF1Evaluator()
    sc = _scenario([])
    s = ev.evaluate(sc, {"seizure_events": []})
    assert s.value == 1.0
    assert s.details["n_gold"] == 0


def test_false_alarm_evaluator_per_hour():
    ev = SeizureFalseAlarmEvaluator()
    sc = _scenario([], duration_sec=1800.0)  # 0.5 h
    s = ev.evaluate(sc, {"seizure_events": [{"onset_sec": 10, "offset_sec": 20}]})
    assert s.value == 1.0
    assert abs(s.details["false_alarms_per_hour"] - 2.0) < 1e-9  # 1 FA / 0.5h


def test_false_alarm_evaluator_zero_when_correct():
    ev = SeizureFalseAlarmEvaluator()
    sc = _scenario([[100, 140]])
    s = ev.evaluate(sc, {"seizure_events": [{"onset_sec": 110, "offset_sec": 130}]})
    assert s.value == 0.0


def test_reversed_interval_normalised():
    # offset < onset should be auto-sorted, still matches
    ev = SeizureEventF1Evaluator()
    sc = _scenario([[100, 140]])
    s = ev.evaluate(sc, {"seizure_events": [{"onset_sec": 150, "offset_sec": 95}]})
    assert s.value == 1.0


# ── Corpus-level seizure metric aggregation ──────────────────────────

def test_seizure_record_positive_and_negative():
    # positive record → dice set, FAR/h None
    r = seizure_record([(100, 140)], [(110, 130)], duration_sec=3600.0)
    assert r["has_seizure"] and r["pred_positive"]
    assert r["dice"] is not None and r["false_alarms_per_hour"] is None
    # seizure-free record with 2 false alarms over 1h → FAR/h == 2
    r0 = seizure_record([], [(10, 20), (50, 60)], duration_sec=3600.0)
    assert (not r0["has_seizure"]) and r0["pred_positive"]
    assert r0["dice"] is None and abs(r0["false_alarms_per_hour"] - 2.0) < 1e-9


def test_seizure_record_from_details_roundtrip():
    d = seizure_record_from_details({
        "seizure_dice": {"has_seizure": True, "n_pred": 2, "dice": 0.5},
        "seizure_false_alarms": {"false_alarms_per_hour": 4.0},
    })
    assert d == {"has_seizure": True, "pred_positive": True,
                 "dice": 0.5, "false_alarms_per_hour": None}
    d0 = seizure_record_from_details({
        "seizure_dice": {"has_seizure": False, "n_pred": 3, "dice": 0.0},
        "seizure_false_alarms": {"false_alarms_per_hour": 3.0},
    })
    assert d0 == {"has_seizure": False, "pred_positive": True,
                  "dice": None, "false_alarms_per_hour": 3.0}


def test_aggregate_seizure_metrics_perfect():
    # 2 positives correctly localised, 2 negatives silent → all perfect
    recs = [
        seizure_record([(100, 140)], [(100, 140)], 3600.0),
        seizure_record([(200, 260)], [(200, 260)], 3600.0),
        seizure_record([], [], 3600.0),
        seizure_record([], [], 3600.0),
    ]
    a = aggregate_seizure_metrics(recs)
    assert a["sensitivity"] == 1.0
    assert abs(a["dice_at_seizure"] - 1.0) < 1e-9
    assert a["macro_f1"] == 1.0
    assert a["far_per_hour"] == 0.0
    assert a["specificity"] == 1.0
    assert a["confusion"] == {"tp": 2, "fn": 0, "fp": 0, "tn": 2}


def test_aggregate_seizure_metrics_always_negative():
    # model that never predicts: Sens/Dice = 0, but Spec = 1 and, on a
    # balanced set, Macro-F1 = 0.5*(0 + 1) wait -> no_seizure F1 depends on set
    recs = [seizure_record([(1, 5)], [], 3600.0),   # positive, missed
            seizure_record([(1, 5)], [], 3600.0),   # positive, missed
            seizure_record([], [], 3600.0),         # negative, correct
            seizure_record([], [], 3600.0)]         # negative, correct
    a = aggregate_seizure_metrics(recs)
    assert a["sensitivity"] == 0.0
    assert a["dice_at_seizure"] == 0.0
    assert a["far_per_hour"] == 0.0
    assert a["specificity"] == 1.0
    # seizure-F1 = 0 (tp=0); no_seizure-F1 = 2*2/(2*2+2) = 4/6 = 0.6667
    assert abs(a["per_class_f1"]["seizure"] - 0.0) < 1e-9
    assert abs(a["per_class_f1"]["no_seizure"] - (2 / 3)) < 1e-9
    assert abs(a["macro_f1"] - (1 / 3)) < 1e-9


def test_aggregate_seizure_metrics_over_caller():
    # over-caller: catches both seizures (Sens=1) but false-alarms both
    # negatives (Spec=0, FAR/h high) → Macro-F1 penalised vs Sensitivity.
    recs = [seizure_record([(1, 5)], [(1, 5)], 3600.0),
            seizure_record([(1, 5)], [(1, 5)], 3600.0),
            seizure_record([], [(1, 5)], 3600.0),
            seizure_record([], [(1, 5)], 3600.0)]
    a = aggregate_seizure_metrics(recs)
    assert a["sensitivity"] == 1.0        # perfect recall
    assert a["specificity"] == 0.0        # every negative false-alarmed
    assert a["far_per_hour"] == 1.0       # 1 FA / 1h each
    # seizure-F1 = 2*2/(2*2+2) = 0.6667 ; no_seizure-F1 = 0 → macro 0.3333
    assert abs(a["macro_f1"] - (1 / 3)) < 1e-9
