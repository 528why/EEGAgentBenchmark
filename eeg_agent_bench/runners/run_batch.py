"""Batch runner — runs multiple scenarios with progress tracking."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from eeg_agent_bench.agents.base import AgentAdapter
from eeg_agent_bench.config import RolloutConfig
from eeg_agent_bench.core.result import ResultWriter
from eeg_agent_bench.envs.base import Environment
from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.runners.run_scenario import run_scenario
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import EvalResult, Scenario

logger = logging.getLogger(__name__)


def run_batch(
    scenarios: list[Scenario],
    agent: AgentAdapter,
    env: Environment,
    tool_registry: ToolRegistry,
    evaluators: list[BaseEvaluator],
    task_message_builder,
    rollout_config: RolloutConfig | None = None,
    output_base_dir: str = "workspace/outputs",
    run_id: str | None = None,
    run_config: dict[str, Any] | None = None,
    redact_gold: bool = False,
    official_scenarios: list[Scenario] | None = None,
) -> list[EvalResult]:
    """Run a batch of scenarios and write results.

    Args:
        scenarios: List of scenarios to evaluate.
        agent: Agent adapter instance.
        env: Environment instance.
        tool_registry: Tool registry.
        evaluators: List of evaluators.
        task_message_builder: Callable(scenario, tool_specs) -> messages.
        rollout_config: Rollout parameters.
        output_base_dir: Base directory for outputs.
        run_id: Optional run identifier. Auto-generated if not provided.
        run_config: Optional config dict to log (no API keys).
        redact_gold: If True, strip gold labels from score_details in
            output files (safe for leaderboard / public distribution).

    Returns:
        List of EvalResults.
    """
    config = rollout_config or RolloutConfig()
    run_id = run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_base_dir) / run_id

    writer = ResultWriter(output_dir, redact_gold=redact_gold)

    # Write run config
    if run_config:
        writer.write_run_config(run_config)

    results: list[EvalResult] = []
    total = len(scenarios)

    for idx, scenario in enumerate(scenarios, 1):
        logger.info(f"[{idx}/{total}] Running scenario: {scenario.scenario_id}")

        try:
            # Build initial messages via task prompt builder
            tool_specs = tool_registry.list_for_scenario(scenario)
            initial_messages = task_message_builder(scenario, tool_specs)

            result, trajectory = run_scenario(
                scenario=scenario,
                agent=agent,
                env=env,
                tool_registry=tool_registry,
                evaluators=evaluators,
                initial_messages=initial_messages,
                rollout_config=config,
            )
        except Exception as e:
            logger.error(f"[{scenario.scenario_id}] Scenario failed: {e}")
            result = EvalResult(
                scenario_id=scenario.scenario_id,
                status="failed",
                error=str(e),
                invalid=True,
            )
            trajectory = []

        # Write results
        writer.write_result(result)
        if trajectory:
            writer.write_trajectory(scenario.scenario_id, trajectory)

        results.append(result)

        # Progress log
        score_str = ", ".join(f"{s.metric}={s.value:.2f}" for s in result.scores)
        logger.info(f"  Status: {result.status} | Scores: {score_str or 'N/A'}")

    # Write aggregate outputs
    writer.write_overall(results, run_config)
    writer.write_failed_cases(results)

    # Official metrics are corpus-level and must be recomputed from the full
    # prediction set and answer key. Never average cached per-item scores.
    if official_scenarios is not None:
        from eeg_agent_bench.reporting.official import (
            OfficialMetricError,
            aggregate_task,
            read_runs,
        )

        try:
            summary = aggregate_task(official_scenarios, read_runs(writer.runs_path))
        except OfficialMetricError as exc:
            summary = {
                "schema": "eegagentbench/official-task-summary/v1",
                "official": False,
                "error": str(exc),
            }
        writer.write_task_summary(summary)

    logger.info(f"Batch complete. Results written to: {output_dir}")
    return results
