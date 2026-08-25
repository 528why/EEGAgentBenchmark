"""Scenario loading and management."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from eeg_agent_bench.types import Scenario

logger = logging.getLogger(__name__)


def load_scenarios(
    path: str | Path,
    answer_key_path: str | Path | None = None,
    runtime_manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> list[Scenario]:
    """Load scenarios from a JSONL file.

    Args:
        path: Path to scenario JSONL (may or may not contain gold_private).
        answer_key_path: Optional separate answer-key JSONL.  When supplied,
            gold is merged by ``scenario_id`` and any gold already embedded
            in the scenario file is **overwritten** by the answer key.
        runtime_manifest_path: Optional private runtime mapping from scenario
            and record IDs to release-relative signal paths. These paths are
            attached only in memory and are never added to model prompts.
        data_root: Root used to resolve relative runtime paths. Defaults to
            the repository root inferred from ``benchmark/runtime/*.jsonl``.

    Returns:
        List of Scenario objects with gold_private populated (if available).
    """
    path = Path(path)
    scenarios: list[Scenario] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scenarios.append(Scenario.from_dict(d))

    # Merge separate answer key if provided
    if answer_key_path is not None:
        ak_path = Path(answer_key_path)
        if ak_path.exists():
            gold_map: dict[str, dict] = {}
            with open(ak_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    sid = entry.get("scenario_id", "")
                    gold = entry.get("gold", {})
                    if sid and gold:
                        gold_map[sid] = gold
            for s in scenarios:
                if s.scenario_id in gold_map:
                    s.gold_private = gold_map[s.scenario_id]
            logger.info(
                "Merged answer key: %d/%d scenarios received gold.",
                sum(1 for s in scenarios if s.gold_private),
                len(scenarios),
            )
        else:
            logger.warning("Answer key path does not exist: %s", ak_path)

    if runtime_manifest_path is not None:
        runtime_path = Path(runtime_manifest_path)
        if not runtime_path.exists():
            raise FileNotFoundError(f"Runtime manifest does not exist: {runtime_path}")
        if data_root is None:
            try:
                resolved_data_root = runtime_path.resolve().parents[2]
            except IndexError as exc:
                raise ValueError(
                    f"Cannot infer repository root from runtime manifest: {runtime_path}"
                ) from exc
        else:
            resolved_data_root = Path(data_root).resolve()

        runtime_map: dict[str, dict[str, str]] = {}
        with runtime_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sid = str(entry.get("scenario_id", ""))
                if not sid or sid in runtime_map:
                    raise ValueError(f"Invalid or duplicate runtime scenario_id: {sid!r}")
                runtime_map[sid] = {
                    str(record["record_id"]): str(record["data_path"])
                    for record in entry.get("records", [])
                }

        for scenario in scenarios:
            record_paths = runtime_map.get(scenario.scenario_id)
            if record_paths is None:
                raise ValueError(
                    f"Runtime manifest has no entry for {scenario.scenario_id}"
                )
            for record in scenario.records:
                if record.record_id not in record_paths:
                    raise ValueError(
                        f"Runtime manifest has no path for "
                        f"{scenario.scenario_id}/{record.record_id}"
                    )
                runtime_value = Path(record_paths[record.record_id])
                record.data_path = str(
                    runtime_value
                    if runtime_value.is_absolute()
                    else resolved_data_root / runtime_value
                )

    return scenarios


def save_scenarios(scenarios: list[Scenario], path: str | Path) -> None:
    """Save scenarios to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def split_public_private(scenario: Scenario) -> tuple[dict, dict]:
    """Split scenario into public (no gold) and private answer key.

    The public dict is guaranteed to contain **no** ``gold_private`` key.
    An assertion guards against accidental re-insertion.
    """
    d = scenario.to_dict()
    gold = d.pop("gold_private", {})
    # Belt-and-suspenders: make sure the key is truly absent
    assert "gold_private" not in d, "gold_private leaked into public scenario dict"
    public = d
    private = {
        "scenario_id": scenario.scenario_id,
        "gold": gold,
    }
    return public, private
