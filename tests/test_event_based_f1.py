"""Tests for the SzCORE-style event-based seizure metric."""

from __future__ import annotations

import pytest

from eeg_agent_bench.evaluators.event_based_f1 import (
    EventBasedParameters,
    aggregate_event_based_metrics,
    merge_close_events,
    score_event_based_record,
    split_long_events,
)


def test_start_and_end_tolerance_detect_early_or_late_predictions():
    early = score_event_based_record([[100, 140]], [[80, 90]], 3600)
    late = score_event_based_record([[100, 140]], [[170, 180]], 3600)
    assert (early["tp"], early["fp"], early["fn"]) == (1, 0, 0)
    assert (late["tp"], late["fp"], late["fn"]) == (1, 0, 0)
    assert early["event_based_f1"] == 1.0
    assert late["event_based_f1"] == 1.0


def test_prediction_outside_tolerance_is_false_positive_and_miss():
    result = score_event_based_record([[100, 140]], [[210, 220]], 3600)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["event_based_f1"] == 0.0


def test_gap_strictly_below_ninety_seconds_merges():
    assert merge_close_events([(100, 120), (209.9, 220)], 90) == [(100, 220)]
    assert merge_close_events([(100, 120), (210, 220)], 90) == [
        (100, 120),
        (210, 220),
    ]


def test_long_event_is_split_into_five_minute_segments():
    assert split_long_events([(0, 720)], 300) == [
        (0, 300),
        (300, 600),
        (600, 720),
    ]


def test_one_prediction_can_detect_multiple_reference_events():
    parameters = EventBasedParameters(min_duration_between_events_sec=0.0)
    result = score_event_based_record(
        [[100, 140], [300, 340]],
        [[120, 320]],
        3600,
        parameters,
    )
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["event_based_f1"] == 1.0


def test_multiple_predictions_within_detected_window_do_not_add_false_positives():
    result = score_event_based_record(
        [[100, 140]],
        [[80, 90], [120, 130], [170, 180]],
        3600,
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert result["precision"] == 1.0


def test_prediction_is_clipped_and_invalid_intervals_are_reported():
    result = score_event_based_record(
        [[0, 10]],
        [[-5, 5], [20, 20], [float("nan"), 30]],
        60,
    )
    assert result["n_pred_input"] == 1
    assert result["tp"] == 1
    assert len(result["prediction_warnings"]) == 2


def test_subject_average_differs_from_pooled_f1():
    records = [
        {"subject_id": "s1", "tp": 1, "fp": 0, "fn": 0, "duration_sec": 10},
        {"subject_id": "s2", "tp": 0, "fp": 9, "fn": 1, "duration_sec": 90},
    ]
    result = aggregate_event_based_metrics(records)
    assert result["subject_average"]["event_based_f1"] == pytest.approx(0.5)
    assert result["pooled"]["event_based_f1"] == pytest.approx(1 / 6)


def test_seizure_free_silent_record_has_no_record_level_f1():
    result = score_event_based_record([], [], 3600)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 0)
    assert result["event_based_f1"] is None
