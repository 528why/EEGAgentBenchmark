#!/usr/bin/env python3
"""Export the frozen benchmark into public, gold, and runtime layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASKS = {
    "T1": ("t1_knowledge", "knowledge_qa", 38, 1000),
    "T2": ("t2_artifact", "artifact_identification", 300, 1000),
    "T3": ("t3_screening", "epilepsy_screening", 188, 1000),
    "T4": ("t4_cohort", "dementia_cohort_classification", 69, 1000),
    "T5": ("t5_seizure", "seizure_detection", 280, 1000),
    "T6": ("t6_sleep", "sleep_staging", 197, 1000),
}

OFFICIAL_TOOLS = [
    "get_recording_info",
    "get_channel_list",
    "compute_noise_metrics",
    "compute_psd",
    "compute_band_power",
    "detect_transients",
    "compute_temporal_features",
    "compute_asymmetry",
    "compute_channel_correlation",
    "compute_windowed_features",
]

SAFE_METADATA = {
    "channels",
    "channel_names",
    "eeg_channels",
    "n_channels",
    "sampling_rate",
    "n_samples",
    "duration_sec",
    "montage",
    "reference",
    "age",
    "gender",
    "sex",
    "night",
    "source",
    "epoch_sec",
    "n_epochs_total",
}

METRICS = {
    "T1": {"primary": "accuracy", "secondary": []},
    "T2": {"primary": "macro_f1", "secondary": ["accuracy"]},
    "T3": {"primary": "macro_f1", "secondary": ["accuracy"]},
    "T4": {"primary": "macro_f1", "secondary": ["accuracy"]},
    "T5": {"primary": "event_f1", "secondary": ["dice_s"]},
    "T6": {"primary": "macro_f1", "secondary": ["cohen_kappa"]},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (metadata or {}).items() if key in SAFE_METADATA}


def release_data_path(source: str) -> str:
    prefixes = {
        "workspace/processed/eegdenoisenet/": "data/eegdenoisenet/",
        "workspace/raw_data/bonn/": "data/bonn/",
        "workspace/raw_data/ds004504-1.0.8/": "data/ds004504/",
        "workspace/raw_data/chbmit/1.0.0/": "data/chbmit/",
        "workspace/processed/sleep_edfx_v3/": "data/sleep_edfx/processed/",
        "workspace/raw_data/sleep-edfx/1.0.0/": "data/sleep_edfx/raw/",
    }
    for old, new in prefixes.items():
        if source.startswith(old):
            return new + source[len(old):]
    raise ValueError(f"Unsupported frozen data path: {source}")


def public_scenario(row: dict[str, Any], display_id: str, max_turns: int) -> dict[str, Any]:
    records = []
    for record in row.get("records", []):
        clean = {
            "record_id": record["record_id"],
            "role": record.get("role", "target"),
            "metadata": clean_metadata(record.get("metadata")),
        }
        if record.get("order") is not None:
            clean["order"] = record["order"]
        records.append(clean)

    input_data = dict(row.get("input", {}))
    input_data.pop("orig_split", None)
    input_data.pop("_orig_split", None)
    if "metadata" in input_data:
        input_data["metadata"] = clean_metadata(input_data.get("metadata"))

    return {
        "scenario_id": row["scenario_id"],
        "task_id": display_id,
        "task_name": row.get("task_name", ""),
        "dataset": row["dataset"],
        "dataset_version": row.get("dataset_version", ""),
        "records": records,
        "subjects": row.get("subjects", []),
        "input": input_data,
        "access": {
            "mode": row.get("access", {}).get("mode", "metadata_with_tools"),
            "allowed_tools": [] if display_id == "T1" else OFFICIAL_TOOLS,
            "forward_only": bool(row.get("access", {}).get("forward_only", False)),
            "max_turns": max_turns,
        },
        "expected_output_schema": row.get("expected_output_schema", {}),
        "evaluator": METRICS[display_id],
        "release": {"split": "evaluation", "include_gold_in_release": False},
    }


def fill_missing_duration(row: dict[str, Any], source_root: Path) -> None:
    """Freeze EDF-header duration when legacy scenario metadata omitted it."""
    records = row.get("records", [])
    if len(records) != 1 or records[0].get("metadata", {}).get("duration_sec") is not None:
        return
    import mne

    source_path = source_root / records[0]["data_path"]
    raw = mne.io.read_raw_edf(source_path, preload=False, verbose=False)
    duration_sec = float(raw.n_times / raw.info["sfreq"])
    records[0].setdefault("metadata", {})["duration_sec"] = duration_sec
    row.setdefault("input", {}).setdefault("metadata", {})["duration_sec"] = duration_sec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    exported: dict[str, Any] = {}

    for display_id, (source_dir, task_key, expected_count, max_turns) in TASKS.items():
        source = source_root / "datasets_v3" / source_dir / f"{source_dir}.eval.jsonl"
        rows = read_jsonl(source)
        if len(rows) != expected_count:
            raise ValueError(f"{display_id}: expected {expected_count} rows, found {len(rows)}")
        if len({row["scenario_id"] for row in rows}) != len(rows):
            raise ValueError(f"{display_id}: duplicate scenario_id")
        if display_id == "T5":
            for row in rows:
                fill_missing_duration(row, source_root)

        public_rows = [public_scenario(row, display_id, max_turns) for row in rows]
        answer_rows = [
            {"scenario_id": row["scenario_id"], "gold": row.get("gold_private", {})}
            for row in rows
        ]
        runtime_rows = []
        for row in rows:
            runtime_records = []
            for record in row.get("records", []):
                runtime_records.append(
                    {
                        "record_id": record["record_id"],
                        "data_path": release_data_path(record["data_path"]),
                    }
                )
            runtime_rows.append({"scenario_id": row["scenario_id"], "records": runtime_records})

        scenario_path = output_root / "benchmark" / "scenarios" / f"{display_id}.jsonl"
        answer_path = output_root / "benchmark" / "answer_keys" / f"{display_id}.jsonl"
        runtime_path = output_root / "benchmark" / "runtime" / f"{display_id}.jsonl"
        write_jsonl(scenario_path, public_rows)
        write_jsonl(answer_path, answer_rows)
        write_jsonl(runtime_path, runtime_rows)

        exported[display_id] = {
            "task_key": task_key,
            "count": len(rows),
            "scenario_file": str(scenario_path.relative_to(output_root)),
            "answer_key": str(answer_path.relative_to(output_root)),
            "runtime_manifest": str(runtime_path.relative_to(output_root)),
        }

    print(f"Exported {sum(task['count'] for task in exported.values())} scenarios")


if __name__ == "__main__":
    main()
