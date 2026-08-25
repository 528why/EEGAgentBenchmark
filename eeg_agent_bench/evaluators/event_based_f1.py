"""SzCORE-style event-based scoring for seizure detection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable


Interval = tuple[float, float]


@dataclass(frozen=True)
class EventBasedParameters:
    """Parameters used by the SzCORE seizure event-scoring configuration."""

    tolerance_start_sec: float = 30.0
    tolerance_end_sec: float = 60.0
    min_overlap_sec: float = 0.0
    min_duration_between_events_sec: float = 90.0
    max_event_duration_sec: float = 300.0
    sampling_frequency_hz: float = 10.0


DEFAULT_EVENT_BASED_PARAMETERS = EventBasedParameters()


def _event_bounds(event: Any) -> tuple[Any, Any]:
    if isinstance(event, dict):
        onset = event.get(
            "onset_sec",
            event.get("onset", event.get("start_sec", event.get("start"))),
        )
        offset = event.get(
            "offset_sec",
            event.get("offset", event.get("end_sec", event.get("end"))),
        )
        return onset, offset
    if isinstance(event, (list, tuple)) and len(event) >= 2:
        return event[0], event[1]
    raise ValueError("unsupported interval format")


def parse_event_intervals(
    raw_events: Iterable[Any] | None,
    duration_sec: float,
    *,
    sampling_frequency_hz: float = 10.0,
) -> tuple[list[Interval], list[str]]:
    """Parse, clip, and quantize event intervals for event-based scoring."""

    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be finite and positive")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")

    intervals: list[Interval] = []
    warnings: list[str] = []
    quantum = 1.0 / sampling_frequency_hz
    quantized_duration = round(duration_sec / quantum) * quantum

    for index, event in enumerate(raw_events or []):
        try:
            onset_raw, offset_raw = _event_bounds(event)
            onset = float(onset_raw)
            offset = float(offset_raw)
            if not math.isfinite(onset) or not math.isfinite(offset):
                raise ValueError("non-finite boundary")
            if offset < onset:
                onset, offset = offset, onset
                warnings.append(f"event[{index}] boundaries swapped")

            onset = min(max(onset, 0.0), duration_sec)
            offset = min(max(offset, 0.0), duration_sec)
            onset = round(onset / quantum) * quantum
            offset = round(offset / quantum) * quantum
            onset = min(max(onset, 0.0), quantized_duration)
            offset = min(max(offset, 0.0), quantized_duration)
            if offset <= onset:
                raise ValueError("empty interval after clipping and quantization")
            intervals.append((onset, offset))
        except (TypeError, ValueError, OverflowError) as exc:
            warnings.append(f"event[{index}] dropped: {exc}")

    intervals.sort(key=lambda interval: (interval[0], interval[1]))
    return intervals, warnings


def merge_close_events(
    intervals: Iterable[Interval],
    min_duration_between_events_sec: float,
) -> list[Interval]:
    """Merge events separated by a gap strictly smaller than the threshold."""

    merged: list[Interval] = []
    for onset, offset in sorted(intervals):
        if not merged:
            merged.append((onset, offset))
            continue
        previous_onset, previous_offset = merged[-1]
        if onset - previous_offset < min_duration_between_events_sec:
            merged[-1] = (previous_onset, max(previous_offset, offset))
        else:
            merged.append((onset, offset))
    return merged


def split_long_events(
    intervals: Iterable[Interval],
    max_event_duration_sec: float,
) -> list[Interval]:
    """Split events into consecutive segments no longer than the maximum."""

    if max_event_duration_sec <= 0:
        raise ValueError("max_event_duration_sec must be positive")

    split: list[Interval] = []
    for onset, offset in intervals:
        cursor = onset
        while offset - cursor > max_event_duration_sec:
            split.append((cursor, cursor + max_event_duration_sec))
            cursor += max_event_duration_sec
        if offset > cursor:
            split.append((cursor, offset))
    return split


def preprocess_events(
    intervals: Iterable[Interval],
    parameters: EventBasedParameters = DEFAULT_EVENT_BASED_PARAMETERS,
) -> list[Interval]:
    """Apply SzCORE event merging followed by long-event splitting."""

    merged = merge_close_events(
        intervals,
        parameters.min_duration_between_events_sec,
    )
    return split_long_events(merged, parameters.max_event_duration_sec)


def _extend_reference_events(
    intervals: Iterable[Interval],
    duration_sec: float,
    parameters: EventBasedParameters,
) -> list[Interval]:
    return [
        (
            max(0.0, onset - parameters.tolerance_start_sec),
            min(duration_sec, offset + parameters.tolerance_end_sec),
        )
        for onset, offset in intervals
    ]


def _overlap_duration(first: Interval, second: Interval) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _matches(
    first: Interval,
    second: Interval,
    min_overlap_sec: float,
) -> bool:
    overlap = _overlap_duration(first, second)
    return overlap > min_overlap_sec


def metrics_from_event_counts(
    tp: int,
    fp: int,
    fn: int,
    duration_sec: float,
) -> dict[str, float | None]:
    """Compute SzCORE-style metrics from accumulated event counts."""

    sensitivity_denominator = tp + fn
    precision_denominator = tp + fp
    return {
        "sensitivity": (
            tp / sensitivity_denominator if sensitivity_denominator else None
        ),
        "precision": tp / precision_denominator if precision_denominator else None,
        "event_based_f1": event_based_f1(tp, fp, fn),
        "fp_per_24h": (
            fp / (duration_sec / 86400.0) if duration_sec > 0 else None
        ),
    }


def event_based_f1(tp: int, fp: int, fn: int) -> float | None:
    """Return event F1 from accumulated TP, FP, and FN counts."""

    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def score_event_based_record(
    gold_events: Iterable[Any] | None,
    predicted_events: Iterable[Any] | None,
    duration_sec: float,
    parameters: EventBasedParameters = DEFAULT_EVENT_BASED_PARAMETERS,
) -> dict[str, Any]:
    """Score one recording with SzCORE-style many-to-many event matching."""

    gold, gold_warnings = parse_event_intervals(
        gold_events,
        duration_sec,
        sampling_frequency_hz=parameters.sampling_frequency_hz,
    )
    if gold_warnings:
        raise ValueError("Invalid gold annotation: " + "; ".join(gold_warnings))
    predicted, prediction_warnings = parse_event_intervals(
        predicted_events,
        duration_sec,
        sampling_frequency_hz=parameters.sampling_frequency_hz,
    )

    processed_gold = preprocess_events(gold, parameters)
    processed_predicted = preprocess_events(predicted, parameters)
    extended_gold = _extend_reference_events(
        processed_gold,
        duration_sec,
        parameters,
    )

    gold_detected = [
        any(
            _matches(reference, prediction, parameters.min_overlap_sec)
            for prediction in processed_predicted
        )
        for reference in extended_gold
    ]
    prediction_matched = [
        any(
            _matches(prediction, reference, parameters.min_overlap_sec)
            for reference in extended_gold
        )
        for prediction in processed_predicted
    ]

    tp = sum(gold_detected)
    fn = len(processed_gold) - tp
    fp = len(processed_predicted) - sum(prediction_matched)
    metrics = metrics_from_event_counts(tp, fp, fn, duration_sec)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duration_sec": duration_sec,
        "n_gold_input": len(gold),
        "n_pred_input": len(predicted),
        "n_gold_processed": len(processed_gold),
        "n_pred_processed": len(processed_predicted),
        "prediction_warnings": prediction_warnings,
        **metrics,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _population_std(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / len(finite))


def aggregate_event_based_metrics(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate record counts within subjects, then average subjects equally."""

    subject_counts: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "duration_sec": 0.0}
    )
    for record in records:
        subject_id = str(record["subject_id"])
        counts = subject_counts[subject_id]
        counts["tp"] += int(record["tp"])
        counts["fp"] += int(record["fp"])
        counts["fn"] += int(record["fn"])
        counts["duration_sec"] += float(record["duration_sec"])

    subjects: dict[str, dict[str, Any]] = {}
    for subject_id, counts in sorted(subject_counts.items()):
        metrics = metrics_from_event_counts(
            int(counts["tp"]),
            int(counts["fp"]),
            int(counts["fn"]),
            float(counts["duration_sec"]),
        )
        subjects[subject_id] = {**counts, **metrics}

    metric_names = ("sensitivity", "precision", "event_based_f1", "fp_per_24h")
    subject_average: dict[str, Any] = {"n_subjects": len(subjects)}
    for metric_name in metric_names:
        values = [subject[metric_name] for subject in subjects.values()]
        subject_average[metric_name] = _mean(values)
        subject_average[f"{metric_name}_std"] = _population_std(values)
        subject_average[f"{metric_name}_n"] = sum(value is not None for value in values)

    pooled_tp = sum(int(counts["tp"]) for counts in subject_counts.values())
    pooled_fp = sum(int(counts["fp"]) for counts in subject_counts.values())
    pooled_fn = sum(int(counts["fn"]) for counts in subject_counts.values())
    pooled_duration = sum(
        float(counts["duration_sec"]) for counts in subject_counts.values()
    )
    pooled = {
        "tp": pooled_tp,
        "fp": pooled_fp,
        "fn": pooled_fn,
        "duration_sec": pooled_duration,
        **metrics_from_event_counts(
            pooled_tp,
            pooled_fp,
            pooled_fn,
            pooled_duration,
        ),
    }

    return {
        "parameters": asdict(DEFAULT_EVENT_BASED_PARAMETERS),
        "subject_average": subject_average,
        "pooled": pooled,
        "subjects": subjects,
    }


__all__ = [
    "DEFAULT_EVENT_BASED_PARAMETERS",
    "EventBasedParameters",
    "aggregate_event_based_metrics",
    "event_based_f1",
    "merge_close_events",
    "metrics_from_event_counts",
    "parse_event_intervals",
    "preprocess_events",
    "score_event_based_record",
    "split_long_events",
]
