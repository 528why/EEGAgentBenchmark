"""Base evaluator interface."""

from __future__ import annotations

import abc
from typing import Any

from eeg_agent_bench.types import EvalScore, Scenario

# Keys in EvalScore.details that contain gold information and must be
# stripped before writing to leaderboard / public outputs.
GOLD_DETAIL_KEYS = frozenset({
    "gold", "gold_label", "gold_tags",
})


def redact_details(details: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *details* with gold-bearing keys removed."""
    return {k: v for k, v in details.items() if k not in GOLD_DETAIL_KEYS}


class BaseEvaluator(abc.ABC):
    """Abstract base for evaluators.

    Evaluators read agent prediction and private gold.
    They never expose gold to the agent runtime or tool environment.
    """

    @property
    @abc.abstractmethod
    def metric_name(self) -> str:
        """Unique metric identifier."""
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        scenario: Scenario,
        prediction: dict[str, Any],
    ) -> EvalScore:
        """Score a single prediction against gold.

        Args:
            scenario: Full scenario (evaluator reads gold_private).
            prediction: Agent's structured output.

        Returns:
            EvalScore with metric value and details.
        """
        ...
