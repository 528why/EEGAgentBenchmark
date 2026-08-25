#!/usr/bin/env python3
"""Validate frozen benchmark integrity and public-release boundaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"T1": 38, "T2": 300, "T3": 188, "T4": 69, "T5": 280, "T6": 197}
OFFICIAL_TOOLS = {
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
}
OFFICIAL_MAX_TURNS = 1000
FORBIDDEN_METADATA = {
    "label",
    "label_binary",
    "labels",
    "set_letter",
    "set_name",
    "group",
    "group_code",
    "subject_type",
    "condition",
    "description",
    "recording_type",
    "has_seizure",
    "n_seizures",
    "seizure_events",
    "hypnogram_path",
    "epoch_labels_aasm",
    "answer",
    "cop",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_public_row(row: dict[str, Any], task_id: str) -> None:
    blob = json.dumps(row, ensure_ascii=False)
    if "gold_private" in row or '"data_path"' in blob:
        raise ValueError(f"{task_id}/{row.get('scenario_id')}: private field in public scenario")
    if any(prefix in blob for prefix in ("/mnt/", "/home/", "/root/")):
        raise ValueError(f"{task_id}/{row.get('scenario_id')}: absolute path leak")
    expected_tools = set() if task_id == "T1" else OFFICIAL_TOOLS
    actual_tools = set(row.get("access", {}).get("allowed_tools", []))
    if actual_tools != expected_tools:
        raise ValueError(f"{task_id}/{row.get('scenario_id')}: wrong official tool set")
    if row.get("access", {}).get("max_turns") != OFFICIAL_MAX_TURNS:
        raise ValueError(
            f"{task_id}/{row.get('scenario_id')}: max_turns must be "
            f"{OFFICIAL_MAX_TURNS}"
        )
    for metadata in [
        *(record.get("metadata", {}) for record in row.get("records", [])),
        row.get("input", {}).get("metadata", {}),
    ]:
        leaked = FORBIDDEN_METADATA & set(metadata)
        if leaked:
            raise ValueError(
                f"{task_id}/{row.get('scenario_id')}: label metadata leak {sorted(leaked)}"
            )


def verify_file(path: Path, expected_size: int) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"missing or wrong-size signal: {path}")


def scan_text_release() -> None:
    secret = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|api_key\s*:\s*[^$\s#][^\s#]{7,})")
    internal_host = re.compile(
        r"https?://(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|"
        r"[^\s/'\"]*\.(?:internal|intranet|corp)(?:[/:]|$))"
    )
    organization_fingerprint = re.compile(
        r"(?i)(?:\bcode/[A-Za-z0-9_.-]+|\binternal\s+(?:\w+\s+){0,2}gateway\b|"
        r"\bx-[A-Za-z0-9-]*(?:email|user)\b)"
    )
    roots = [ROOT / "eeg_agent_bench", ROOT / "configs", ROOT / "scripts", ROOT / "docs"]
    for base in roots:
        for path in base.rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or ".pyc" in path.suffixes
                or path.suffix in {".png", ".pdf"}
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if secret.search(text):
                raise ValueError(f"possible embedded secret: {path.relative_to(ROOT)}")
            if internal_host.search(text):
                raise ValueError(f"internal endpoint in release: {path.relative_to(ROOT)}")
            if organization_fingerprint.search(text):
                raise ValueError(
                    f"possible organization fingerprint: {path.relative_to(ROOT)}"
                )
            machine_prefix = "/mnt/" + "tidal-"
            if machine_prefix in text:
                raise ValueError(f"machine-specific path in release: {path.relative_to(ROOT)}")


def main() -> None:
    total = 0
    signal_paths: set[str] = set()
    for task_id, expected_count in EXPECTED.items():
        scenarios = read_jsonl(ROOT / "benchmark/scenarios" / f"{task_id}.jsonl")
        answers = read_jsonl(ROOT / "benchmark/answer_keys" / f"{task_id}.jsonl")
        runtime = read_jsonl(ROOT / "benchmark/runtime" / f"{task_id}.jsonl")
        if not (len(scenarios) == len(answers) == len(runtime) == expected_count):
            raise ValueError(f"{task_id}: scenario/answer/runtime count mismatch")
        scenario_ids = {row["scenario_id"] for row in scenarios}
        if len(scenario_ids) != expected_count:
            raise ValueError(f"{task_id}: duplicate scenario IDs")
        if scenario_ids != {row["scenario_id"] for row in answers}:
            raise ValueError(f"{task_id}: answer-key IDs do not align")
        if scenario_ids != {row["scenario_id"] for row in runtime}:
            raise ValueError(f"{task_id}: runtime IDs do not align")
        for row in scenarios:
            check_public_row(row, task_id)
        for row in runtime:
            for record in row.get("records", []):
                signal_paths.add(record["data_path"])
        total += expected_count

    if total != 1072 or len(signal_paths) != 1034:
        raise ValueError(f"frozen totals are wrong: instances={total}, signals={len(signal_paths)}")

    file_rows = read_jsonl(ROOT / "benchmark/manifests/files.jsonl")
    manifest_paths = {row["data_path"] for row in file_rows}
    if len(file_rows) != 1034 or manifest_paths != signal_paths:
        raise ValueError("signal file manifest does not match runtime manifests")
    for row in file_rows:
        verify_file(ROOT / row["data_path"], int(row["size_bytes"]))

    benchmark = json.loads((ROOT / "benchmark/benchmark.json").read_text(encoding="utf-8"))
    if benchmark["total_instances"] != 1072:
        raise ValueError("benchmark.json total_instances is wrong")
    if set(benchmark["official_tools"]) != OFFICIAL_TOOLS:
        raise ValueError("benchmark.json official tool set is wrong")
    if any(
        task.get("max_turns") != OFFICIAL_MAX_TURNS
        for task in benchmark["tasks"].values()
    ):
        raise ValueError("benchmark.json tasks must all use max_turns=1000")

    scan_text_release()
    print(
        f"Release validation passed: {total} instances, {len(signal_paths)} signals, "
        f"{len(OFFICIAL_TOOLS)} official tools"
    )


if __name__ == "__main__":
    main()
