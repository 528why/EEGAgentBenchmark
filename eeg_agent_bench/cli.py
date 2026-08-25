"""Command-line interface for EEG-AgentBench."""

from __future__ import annotations

import logging
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eeg_agent_bench")


def _companion_path(scenario_path: str | Path, directory: str) -> Path | None:
    scenario = Path(scenario_path).resolve()
    if scenario.parent.name != "scenarios":
        return None
    candidate = scenario.parent.parent / directory / scenario.name
    return candidate if candidate.exists() else None


@click.group()
@click.version_option(version="1.0.0", prog_name="eeg-agent-bench")
def main():
    """EEG-AgentBench: Agentic Capability Evaluation for EEG Analysis."""
    pass


@main.command()
@click.option("--scenario", "-s", required=True, help="Path to scenario JSONL file.")
@click.option("--answer-key", default=None, help="Path to separate answer key JSONL (private gold).")
@click.option("--runtime-manifest", default=None, help="Scenario-to-signal runtime manifest.")
@click.option("--data-root", default=None, help="Root for release-relative signal paths.")
@click.option("--agent", "-a", required=True, help="Path to agent config YAML.")
@click.option("--tools", "-t", default="configs/tools.yaml", help="Path to tools config.")
@click.option("--output", "-o", default="workspace/outputs", help="Output base directory.")
@click.option("--max-turns", default=None, type=int,
              help="Max turns. Defaults to the frozen task contract.")
@click.option("--max-tool-calls", default=None, type=int,
              help="Max tool calls. Defaults to 0 for T1 and 999 for T2-T6.")
@click.option("--max-protocol-retries", default=3, type=int,
              help="Retries for unparseable actions or schema-invalid final answers.")
@click.option("--task", "task_override", default=None, type=str,
              help="Force a specific task_id (e.g. C3-Routine, C3-Cohort). "
                   "If omitted, the task is auto-detected from the first scenario's task_id.")
@click.option("--resume-dir", default=None, type=str,
              help="Explicit run directory to resume into (appends to its runs.jsonl).")
@click.option("--resume/--no-resume", default=False,
              help="Auto-detect latest run directory under --output and resume.")
@click.option("--redact-gold/--no-redact-gold", default=False,
              help="Strip gold labels from score_details in output files.")
@click.option("--official/--non-official", default=True,
              help="Require the complete frozen task and write official metrics.")
def run(scenario: str, answer_key: str | None, runtime_manifest: str | None,
        data_root: str | None, agent: str, tools: str,
        output: str, max_turns: int | None, max_tool_calls: int | None,
        max_protocol_retries: int,
        task_override: str | None,
        resume_dir: str | None, resume: bool, redact_gold: bool,
        official: bool):
    """Run scenarios against an agent.

    The task class and evaluator set are dispatched from the task
    registry based on the ``task_id`` of the first scenario (or the
    explicit ``--task`` flag). Semantic task keys, T1–T6 display IDs, and
    supported historical C/K identifiers are accepted.
    """
    from eeg_agent_bench.agents.base import AgentAdapterFactory
    from eeg_agent_bench.config import AgentConfig, RolloutConfig
    from eeg_agent_bench.core.scenario import load_scenarios
    from eeg_agent_bench.envs.tool_env import ToolEnvironment
    from eeg_agent_bench.runners.resume import filter_pending, load_completed_ids
    from eeg_agent_bench.runners.run_batch import run_batch
    from eeg_agent_bench.tasks.registry import (
        display_task_id,
        get_task_bundle,
        resolve_task_key,
        is_registered,
        list_task_ids,
    )
    from eeg_agent_bench.tools.loader import load_tool_registry
    from eeg_agent_bench.tools.preflight import validate_tool_runtime

    # Ensure adapters are registered
    import eeg_agent_bench.agents.adapters  # noqa: F401

    # Load components
    logger.info(f"Loading scenarios from: {scenario}")
    resolved_answer_key = Path(answer_key) if answer_key else _companion_path(scenario, "answer_keys")
    resolved_runtime = (
        Path(runtime_manifest)
        if runtime_manifest
        else _companion_path(scenario, "runtime")
    )
    if resolved_answer_key is None or not resolved_answer_key.exists():
        raise click.ClickException(
            "Official evaluation requires an answer key. Use --answer-key."
        )
    scenarios = load_scenarios(
        scenario,
        answer_key_path=resolved_answer_key,
        runtime_manifest_path=resolved_runtime,
        data_root=data_root,
    )
    logger.info(f"Loaded {len(scenarios)} scenarios.")

    if not scenarios:
        logger.error("No scenarios loaded; aborting.")
        sys.exit(1)

    # Preserve the complete set for official corpus-level aggregation after a
    # resume filters already-completed scenarios out of the execution queue.
    official_scenarios = list(scenarios)

    # ── Task dispatch via registry ────────────────────────────────
    task_id = task_override or scenarios[0].task_id
    if not is_registered(task_id):
        logger.error(
            f"Task '{task_id}' is not registered. "
            f"Available: {list_task_ids()}"
        )
        sys.exit(1)
    bundle = get_task_bundle(task_id)
    task = bundle.task
    evaluators = bundle.evaluators
    logger.info(
        f"Task: {display_task_id(task_id)} / {resolve_task_key(task_id)} "
        f"({task.task_name}) — "
        f"{len(evaluators)} evaluator(s): "
        f"{[e.metric_name for e in evaluators]}"
    )

    agent_config = AgentConfig.from_yaml(agent)
    agent_adapter = AgentAdapterFactory.create(agent_config)
    logger.info(f"Agent: {agent_config.name} ({agent_config.model})")

    tool_registry = load_tool_registry(tools if Path(tools).exists() else None)
    logger.info(f"Tools: {len(tool_registry.list_names())} registered.")
    try:
        validate_tool_runtime(scenarios, tool_registry)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    env = ToolEnvironment(tool_registry)

    effective_max_turns = max_turns if max_turns is not None else scenarios[0].access.max_turns
    effective_max_tool_calls = (
        max_tool_calls
        if max_tool_calls is not None
        else (0 if display_task_id(task_id) == "T1" else 999)
    )
    rollout_config = RolloutConfig(
        max_turns=effective_max_turns,
        max_tool_calls=effective_max_tool_calls,
        max_protocol_retries=max_protocol_retries,
    )

    # ── Resume support ────────────────────────────────────────────
    # Determine the run directory to resume into (if any).
    resolved_run_id: str | None = None

    if resume_dir:
        # Explicit directory — resume into it directly.
        resume_path = Path(resume_dir)
        if not resume_path.exists():
            logger.error(f"--resume-dir does not exist: {resume_dir}")
            sys.exit(1)
        completed = load_completed_ids(resume_path / "runs.jsonl")
        before = len(scenarios)
        scenarios = filter_pending(scenarios, completed)
        logger.info(f"Resuming {resume_path.name}: {before - len(scenarios)} done, "
                     f"{len(scenarios)} pending.")
        # Reuse the same run directory (run_id = dir name relative to base)
        resolved_run_id = str(resume_path.relative_to(Path(output))) \
            if str(resume_path).startswith(str(Path(output))) \
            else resume_path.name

    elif resume:
        # Auto-detect latest run directory under --output.
        output_path = Path(output)
        if output_path.exists():
            subdirs = sorted(
                [d for d in output_path.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if subdirs:
                latest = subdirs[0]
                completed = load_completed_ids(latest / "runs.jsonl")
                before = len(scenarios)
                scenarios = filter_pending(scenarios, completed)
                logger.info(f"Resuming latest run {latest.name}: {before - len(scenarios)} done, "
                             f"{len(scenarios)} pending.")
                resolved_run_id = latest.name
            else:
                logger.warning("No previous runs found; starting fresh.")
        else:
            logger.warning(f"Output dir {output} does not exist; starting fresh.")

    if not scenarios:
        logger.info("All scenarios already completed. Nothing to do.")
        return

    # ── Run ───────────────────────────────────────────────────────
    run_config = {
        "agent": agent_config.to_log_dict(),
        "scenario_file": scenario,
        "tools_config": tools,
        "rollout": {
            "max_turns": effective_max_turns,
            "max_tool_calls": effective_max_tool_calls,
            "max_protocol_retries": max_protocol_retries,
        },
    }

    results = run_batch(
        scenarios=scenarios,
        agent=agent_adapter,
        env=env,
        tool_registry=tool_registry,
        evaluators=evaluators,
        task_message_builder=lambda s, ts: task.build_messages(s, ts),
        rollout_config=rollout_config,
        output_base_dir=output,
        run_id=resolved_run_id,
        run_config=run_config,
        redact_gold=redact_gold,
        official_scenarios=official_scenarios if official else None,
    )

    # Summary
    completed_count = sum(1 for r in results if r.status == "completed")
    logger.info(f"Done. {completed_count}/{len(results)} scenarios completed.")


@main.command()
@click.option("--scenario", required=True, help="Frozen public scenario JSONL.")
@click.option("--answer-key", default=None, help="Separate answer-key JSONL.")
@click.option("--runs", "runs_path", required=True, help="Completed runs.jsonl.")
@click.option("--output", default=None, help="Output summary.json path.")
def summarize(scenario: str, answer_key: str | None, runs_path: str, output: str | None):
    """Recompute the official paper metric from predictions and gold."""
    from eeg_agent_bench.core.scenario import load_scenarios
    from eeg_agent_bench.reporting.official import aggregate_task, read_runs, write_json

    resolved_answer_key = Path(answer_key) if answer_key else _companion_path(scenario, "answer_keys")
    if resolved_answer_key is None:
        raise click.ClickException("Use --answer-key for official scoring.")
    summary = aggregate_task(
        load_scenarios(scenario, answer_key_path=resolved_answer_key),
        read_runs(runs_path),
    )
    output_path = Path(output) if output else Path(runs_path).parent / "summary.json"
    write_json(output_path, summary)
    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@main.command("run-all")
@click.option("--agent", "agent_path", required=True, help="Agent config YAML.")
@click.option("--output", default="outputs", help="Output root.")
@click.option("--run-name", default=None, help="Stable run directory name for resume.")
@click.option("--tools", default="configs/tools.yaml", help="Official tools config.")
def run_all(agent_path: str, output: str, run_name: str | None, tools: str):
    """Run all six frozen tasks and write the official Overall score."""
    from eeg_agent_bench.reporting.official import aggregate_benchmark, write_json

    root = Path(__file__).resolve().parents[1]
    name = run_name or (
        f"{Path(agent_path).stem}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = Path(output) / name
    run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(1, 7):
        task_id = f"T{index}"
        task_dir = run_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "eeg_agent_bench.cli",
            "run",
            "--scenario",
            str(root / "benchmark" / "scenarios" / f"{task_id}.jsonl"),
            "--agent",
            agent_path,
            "--tools",
            tools,
            "--output",
            str(run_root),
            "--resume-dir",
            str(task_dir),
            "--redact-gold",
        ]
        subprocess.run(command, check=True)
        summary_path = task_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("official") is not True:
            raise click.ClickException(f"{task_id} is incomplete: {summary.get('error')}")
        summaries.append(summary)

    benchmark_summary = aggregate_benchmark(summaries)
    write_json(run_root / "benchmark_summary.json", benchmark_summary)
    click.echo(json.dumps(benchmark_summary, indent=2))


@main.command()
@click.option("--manifest", default="benchmark/manifests/files.jsonl")
def prepare(manifest: str):
    """Verify the staged frozen signal files."""
    root = Path(__file__).resolve().parents[1]
    failures = []
    checked = 0
    with (root / manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            path = root / entry["data_path"]
            if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
                failures.append(entry["data_path"])
                continue
            checked += 1
    if failures:
        raise click.ClickException(
            f"Data verification failed for {len(failures)} file(s)."
        )
    click.echo(f"Verified {checked} signal files (path and size).")


if __name__ == "__main__":
    main()
