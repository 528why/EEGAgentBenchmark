"""Official EEGAgentBench metrics used by the paper.

This module is the only corpus-level aggregation implementation in the
release. It always recomputes metrics from predictions and answer keys; scalar
scores cached in runs.jsonl are deliberately ignored.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eeg_agent_bench.evaluators.artifact_classification import normalise_label
from eeg_agent_bench.evaluators.event_based_f1 import (
    DEFAULT_EVENT_BASED_PARAMETERS,
    aggregate_event_based_metrics,
    score_event_based_record,
)
from eeg_agent_bench.evaluators.hypnogram_metrics import (
    HypnogramCohenKappaEvaluator,
    HypnogramMacroF1Evaluator,
)
from eeg_agent_bench.evaluators.mcq_accuracy import normalise_answer
from eeg_agent_bench.evaluators.seizure_event import (
    _parse_events,
    match_events,
    temporal_dice,
)
from eeg_agent_bench.tasks.registry import display_task_id
from eeg_agent_bench.types import Scenario

EXPECTED_COUNTS = {"T1": 38, "T2": 300, "T3": 188, "T4": 69, "T5": 280, "T6": 197}
PRIMARY_METRIC = {
    "T1": "accuracy",
    "T2": "macro_f1",
    "T3": "macro_f1",
    "T4": "macro_f1",
    "T5": "event_f1",
    "T6": "macro_f1",
}
REFERENCE_CLASSES = {
    "T1": ("A", "B", "C", "D"),
    "T2": ("clean", "ocular_contaminated", "muscle_contaminated"),
    "T3": ("normal", "epileptic"),
    "T4": ("AD", "FTD", "HC"),
}
SUBJECT_PATTERN = re.compile(r"(?:^|[/_])(chb\d+)[a-z]?(?:[/_]|$)", re.IGNORECASE)


class OfficialMetricError(ValueError):
    """Raised when a result cannot be reported as an official score."""


def read_runs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _index_runs(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for run in runs:
        scenario_id = str(run.get("scenario_id", ""))
        if not scenario_id:
            raise OfficialMetricError("runs.jsonl contains a row without scenario_id")
        if scenario_id in indexed:
            raise OfficialMetricError(f"duplicate run for scenario_id={scenario_id}")
        indexed[scenario_id] = run
    return indexed


def _validate_complete(
    task_id: str,
    scenarios: list[Scenario],
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = EXPECTED_COUNTS[task_id]
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != expected:
        raise OfficialMetricError(
            f"{task_id} requires {expected} scenarios, found {len(scenario_ids)}"
        )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise OfficialMetricError(f"{task_id} scenario file contains duplicate IDs")

    scenario_set = set(scenario_ids)
    run_set = set(runs)
    missing = sorted(scenario_set - run_set)
    extra = sorted(run_set - scenario_set)
    noncompleted = sorted(
        scenario_id
        for scenario_id in scenario_set & run_set
        if runs[scenario_id].get("status") != "completed"
    )
    if missing or extra or noncompleted:
        raise OfficialMetricError(
            f"{task_id} result is incomplete: missing={len(missing)}, "
            f"extra={len(extra)}, noncompleted={len(noncompleted)}"
        )
    return {
        "expected": expected,
        "completed": expected,
        "missing": 0,
        "extra": 0,
        "noncompleted": 0,
    }


def _raw_classification_prediction(run: dict[str, Any]) -> Any:
    prediction = run.get("prediction")
    if not isinstance(prediction, dict):
        return None
    return prediction.get("classification")


def _classification_metrics(
    pairs: list[tuple[str, str]],
    reference_classes: tuple[str, ...],
) -> dict[str, Any]:
    """Accuracy and Macro-F1 over the fixed task reference classes.

    Predictions outside ``reference_classes`` and empty predictions map to
    ``__invalid__``. They remain in the denominator and contribute a false
    negative to the gold class, but never create an additional averaged class.
    """

    if not pairs:
        raise OfficialMetricError("classification metric received no examples")
    expected_set = set(reference_classes)
    gold_set = {gold for gold, _ in pairs}
    if gold_set != expected_set:
        raise OfficialMetricError(
            f"gold classes {sorted(gold_set)} do not match task classes "
            f"{list(reference_classes)}"
        )

    columns = (*reference_classes, "__invalid__")
    confusion = {
        gold: {prediction: 0 for prediction in columns} for gold in reference_classes
    }
    invalid_examples = 0
    for gold, prediction in pairs:
        column = prediction if prediction in expected_set else "__invalid__"
        confusion[gold][column] += 1
        invalid_examples += int(column == "__invalid__")

    per_class: dict[str, dict[str, float | int]] = {}
    for label in reference_classes:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in reference_classes if other != label)
        fn = sum(confusion[label][column] for column in columns if column != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    correct = sum(confusion[label][label] for label in reference_classes)
    return {
        "n": len(pairs),
        "reference_classes": list(reference_classes),
        "accuracy": correct / len(pairs),
        "macro_f1": sum(float(per_class[label]["f1"]) for label in reference_classes)
        / len(reference_classes),
        "invalid_predictions": invalid_examples,
        "per_class": per_class,
        "confusion": confusion,
    }


def _aggregate_t1(scenarios: list[Scenario], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    for scenario in scenarios:
        prediction = runs[scenario.scenario_id].get("prediction")
        raw = prediction.get("answer") if isinstance(prediction, dict) else None
        gold = str(scenario.gold_private.get("answer", "")).strip().upper()
        predicted = normalise_answer(raw, scenario.input.get("options"))
        pairs.append((gold, predicted))
    result = _classification_metrics(pairs, REFERENCE_CLASSES["T1"])
    return {
        "accuracy": result["accuracy"],
        "invalid_predictions": result["invalid_predictions"],
        "confusion": result["confusion"],
    }


def _canonical_classification(task_id: str, raw: Any) -> str:
    if task_id == "T2":
        return normalise_label(raw)
    text = "" if raw is None else str(raw).strip()
    by_casefold = {label.casefold(): label for label in REFERENCE_CLASSES[task_id]}
    return by_casefold.get(text.casefold(), "")


def _aggregate_classification(
    task_id: str,
    scenarios: list[Scenario],
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    for scenario in scenarios:
        gold_raw = str(scenario.gold_private.get("label", "")).strip()
        gold = _canonical_classification(task_id, gold_raw)
        predicted = _canonical_classification(
            task_id, _raw_classification_prediction(runs[scenario.scenario_id])
        )
        pairs.append((gold, predicted))
    return _classification_metrics(pairs, REFERENCE_CLASSES[task_id])


def _subject_id(scenario: Scenario) -> str:
    match = SUBJECT_PATTERN.search(scenario.scenario_id)
    if not match:
        raise OfficialMetricError(
            f"cannot recover CHB-MIT subject from {scenario.scenario_id}"
        )
    return match.group(1).lower()


def _aggregate_t5(scenarios: list[Scenario], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    event_records: list[dict[str, Any]] = []
    dice_values: list[float] = []
    missed_positive = 0
    over_alarm_negative = 0
    n_positive = 0
    n_negative = 0
    prediction_warning_count = 0

    for scenario in scenarios:
        if len(scenario.records) != 1:
            raise OfficialMetricError(f"{scenario.scenario_id} must contain one record")
        duration_sec = float(scenario.records[0].metadata.get("duration_sec", 0.0))
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise OfficialMetricError(f"invalid duration for {scenario.scenario_id}")
        gold_events = scenario.gold_private.get("seizure_events", [])
        prediction = runs[scenario.scenario_id].get("prediction")
        predicted_events = (
            prediction.get("seizure_events", []) if isinstance(prediction, dict) else []
        )

        record_score = score_event_based_record(
            gold_events, predicted_events, duration_sec, DEFAULT_EVENT_BASED_PARAMETERS
        )
        prediction_warning_count += len(record_score["prediction_warnings"])
        event_records.append({"subject_id": _subject_id(scenario), **record_score})

        parsed_gold = _parse_events(gold_events)
        parsed_prediction = _parse_events(predicted_events)
        if parsed_gold:
            n_positive += 1
            dice_values.append(temporal_dice(parsed_gold, parsed_prediction))
            raw_match = match_events(parsed_gold, parsed_prediction)
            missed_positive += int(raw_match["tp_gold"] == 0)
        else:
            n_negative += 1
            over_alarm_negative += int(bool(parsed_prediction))

    if n_positive != 140 or n_negative != 140:
        raise OfficialMetricError(
            f"T5 requires 140 positive and 140 negative records, found "
            f"{n_positive}/{n_negative}"
        )
    aggregate = aggregate_event_based_metrics(event_records)
    event_f1 = aggregate["subject_average"]["event_based_f1"]
    if event_f1 is None:
        event_f1 = 0.0
    return {
        "n": len(scenarios),
        "n_subjects": aggregate["subject_average"]["n_subjects"],
        "event_f1": event_f1,
        "dice_s": sum(dice_values) / len(dice_values),
        "sensitivity": aggregate["subject_average"]["sensitivity"],
        "precision": aggregate["subject_average"]["precision"],
        "fp_per_24h": aggregate["subject_average"]["fp_per_24h"],
        "miss_rate": missed_positive / n_positive,
        "over_alarm_rate": over_alarm_negative / n_negative,
        "prediction_warning_count": prediction_warning_count,
        "event_parameters": asdict(DEFAULT_EVENT_BASED_PARAMETERS),
        "subject_metrics": aggregate["subjects"],
        "pooled": aggregate["pooled"],
    }


def _aggregate_t6(scenarios: list[Scenario], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    macro_values: list[float] = []
    kappa_values: list[float] = []
    macro_evaluator = HypnogramMacroF1Evaluator()
    kappa_evaluator = HypnogramCohenKappaEvaluator()
    empty_predictions = 0
    per_record: dict[str, dict[str, float]] = {}

    for scenario in scenarios:
        prediction = runs[scenario.scenario_id].get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        empty_predictions += int(not prediction.get("hypnogram_rle"))
        macro = macro_evaluator.evaluate(scenario, prediction).value
        kappa = kappa_evaluator.evaluate(scenario, prediction).value
        macro_values.append(macro)
        kappa_values.append(kappa)
        per_record[scenario.scenario_id] = {"macro_f1": macro, "cohen_kappa": kappa}

    return {
        "n": len(scenarios),
        "macro_f1": sum(macro_values) / len(macro_values),
        "cohen_kappa": sum(kappa_values) / len(kappa_values),
        "empty_predictions": empty_predictions,
        "per_record": per_record,
    }


def aggregate_task(
    scenarios: list[Scenario],
    runs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    if not scenarios:
        raise OfficialMetricError("cannot aggregate an empty scenario set")
    task_ids = {display_task_id(scenario.task_id) for scenario in scenarios}
    if len(task_ids) != 1:
        raise OfficialMetricError(f"scenario file mixes tasks: {sorted(task_ids)}")
    task_id = task_ids.pop()
    indexed_runs = _index_runs(runs)
    completeness = _validate_complete(task_id, scenarios, indexed_runs)

    if task_id == "T1":
        metrics = _aggregate_t1(scenarios, indexed_runs)
    elif task_id in ("T2", "T3", "T4"):
        metrics = _aggregate_classification(task_id, scenarios, indexed_runs)
    elif task_id == "T5":
        metrics = _aggregate_t5(scenarios, indexed_runs)
    elif task_id == "T6":
        metrics = _aggregate_t6(scenarios, indexed_runs)
    else:
        raise OfficialMetricError(f"unsupported task: {task_id}")

    return {
        "schema": "eegagentbench/official-task-summary/v1",
        "official": True,
        "task_id": task_id,
        "primary_metric": PRIMARY_METRIC[task_id],
        "primary_score": metrics[PRIMARY_METRIC[task_id]],
        "completeness": completeness,
        "metrics": metrics,
    }


def aggregate_benchmark(task_summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summaries = {summary["task_id"]: summary for summary in task_summaries}
    expected = [f"T{i}" for i in range(1, 7)]
    if sorted(summaries) != expected:
        raise OfficialMetricError(
            f"Overall requires T1--T6 summaries, found {sorted(summaries)}"
        )
    if not all(summary.get("official") is True for summary in summaries.values()):
        raise OfficialMetricError("Overall requires six official complete task summaries")
    primary_scores = {
        task_id: float(summaries[task_id]["primary_score"]) for task_id in expected
    }
    return {
        "schema": "eegagentbench/official-benchmark-summary/v1",
        "official": True,
        "overall": sum(primary_scores.values()) / len(primary_scores),
        "primary_scores": primary_scores,
        "definition": (
            "Unweighted mean of T1 accuracy, T2-T4 Macro-F1, "
            "T5 subject-averaged Event-F1, and T6 Macro-F1"
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EXPECTED_COUNTS",
    "PRIMARY_METRIC",
    "REFERENCE_CLASSES",
    "OfficialMetricError",
    "aggregate_benchmark",
    "aggregate_task",
    "read_runs",
    "write_json",
]
