#!/usr/bin/env python3
"""Build a deterministic, class-aware smoke subset from the frozen release."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_ids(task_id: str, answers):
    if task_id in {"T1", "T6"}:
        return [answers[0]["scenario_id"]]
    field = "has_seizure" if task_id == "T5" else "label"
    selected = {}
    for row in answers:
        value = row["gold"].get(field)
        selected.setdefault(str(value), row["scenario_id"])
    return list(selected.values())


def main():
    for index in range(1, 7):
        task_id = f"T{index}"
        scenarios = read_jsonl(ROOT / "benchmark/scenarios" / f"{task_id}.jsonl")
        answers = read_jsonl(ROOT / "benchmark/answer_keys" / f"{task_id}.jsonl")
        runtime = read_jsonl(ROOT / "benchmark/runtime" / f"{task_id}.jsonl")
        wanted = set(select_ids(task_id, answers))
        for directory, rows in (
            ("scenarios", scenarios), ("answer_keys", answers), ("runtime", runtime)
        ):
            write_jsonl(
                ROOT / "benchmark/smoke" / directory / f"{task_id}.jsonl",
                [row for row in rows if row["scenario_id"] in wanted],
            )
    print("Built deterministic smoke subsets under benchmark/smoke")


if __name__ == "__main__":
    main()
