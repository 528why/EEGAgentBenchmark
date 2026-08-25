"""Result management: writing runs.jsonl, overall.json, trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eeg_agent_bench.types import EvalResult, TrajectoryEntry


class ResultWriter:
    """Manages writing evaluation results and trajectories to disk.

    Args:
        output_dir: Directory for all output files.
        redact_gold: If ``True``, strip gold-bearing keys from
            ``score_details`` in ``runs.jsonl`` and ``failed_cases.json``
            so outputs are safe for leaderboard / public distribution.
    """

    def __init__(self, output_dir: str | Path, *, redact_gold: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir = self.output_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self._runs_path = self.output_dir / "runs.jsonl"
        self._redact_gold = redact_gold

    def write_run_config(self, config: dict[str, Any]) -> None:
        """Write run configuration (without API keys)."""
        import yaml
        with open(self.output_dir / "run_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def write_result(self, result: EvalResult) -> None:
        """Append a single scenario result to runs.jsonl."""
        with open(self._runs_path, "a") as f:
            f.write(json.dumps(result.to_dict(redact_gold=self._redact_gold), ensure_ascii=False) + "\n")

    def write_trajectory(self, scenario_id: str, entries: list[TrajectoryEntry]) -> None:
        """Write trajectory for a single scenario."""
        path = self.trajectories_dir / f"{scenario_id}.jsonl"
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def write_overall(self, results: list[EvalResult], run_config: dict[str, Any] | None = None) -> None:
        """Write execution status and usage, not official paper metrics.

        Official task metrics are recomputed from predictions plus the answer
        key and written separately to ``summary.json`` by the batch runner.
        """
        total = len(results)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        exceeded = sum(1 for r in results if r.status == "max_turns_exceeded")
        invalid = sum(1 for r in results if getattr(r, "invalid", False))
        scored = total - invalid  # metric denominator

        # Aggregate scores by metric.  Invalid runs carry no scores, so they
        # are naturally excluded from the metric means computed below.
        metric_values: dict[str, list[float]] = {}
        for r in results:
            for s in r.scores:
                metric_values.setdefault(s.metric, []).append(s.value)

        aggregate_scores = {}
        for metric, values in metric_values.items():
            aggregate_scores[metric] = {
                "mean": sum(values) / len(values) if values else 0,
                "count": len(values),
            }

        # ── Aggregate token usage across all scenarios ────────────
        token_keys = [
            "prompt_tokens", "completion_tokens", "total_tokens",
            "reasoning_tokens",
        ]
        aggregate_usage: dict[str, Any] = {
            "total_tool_calls": sum(r.usage.get("tool_calls", 0) for r in results),
            "total_turns": sum(r.usage.get("turns", 0) for r in results),
            "total_wall_time_sec": round(sum(r.usage.get("wall_time_sec", 0) for r in results), 2),
        }
        for key in token_keys:
            val = sum(r.usage.get(key, 0) for r in results)
            if val > 0:
                aggregate_usage[key] = val

        overall: dict[str, Any] = {
            "total_scenarios": total,
            "completed": completed,
            "failed": failed,
            "max_turns_exceeded": exceeded,
            "invalid": invalid,
            "scored": scored,
            "aggregate_scores": aggregate_scores,
            "aggregate_usage": aggregate_usage,
        }
        if run_config:
            overall["run_config"] = run_config

        with open(self.output_dir / "run_status.json", "w") as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)

    def write_task_summary(self, summary: dict[str, Any]) -> None:
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    @property
    def runs_path(self) -> Path:
        return self._runs_path

    def write_failed_cases(self, results: list[EvalResult]) -> None:
        """Write failed_cases.json for debugging."""
        failed = [
            r.to_dict(redact_gold=self._redact_gold)
            for r in results
            if r.status in ("failed", "max_turns_exceeded")
        ]
        with open(self.output_dir / "failed_cases.json", "w") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
