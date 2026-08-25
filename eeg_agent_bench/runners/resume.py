"""Resume support — skip already-completed scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from eeg_agent_bench.types import Scenario


def load_completed_ids(runs_jsonl_path: str | Path) -> set[str]:
    """Read runs.jsonl and return set of completed scenario IDs."""
    completed: set[str] = set()
    path = Path(runs_jsonl_path)
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("status") == "completed":
                    completed.add(d["scenario_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def filter_pending(
    scenarios: list[Scenario],
    completed_ids: set[str],
) -> list[Scenario]:
    """Filter out scenarios that are already completed."""
    return [s for s in scenarios if s.scenario_id not in completed_ids]
