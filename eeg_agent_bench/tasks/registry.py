"""Task registry with stable semantic keys and presentation-only task IDs.

The benchmark separates four concepts:

- ``task_key``: stable machine identifier (for example ``seizure_detection``),
- ``display_id``: paper/leaderboard label (``T1`` .. ``T6``),
- ``legacy_ids``: identifiers accepted for historical artifacts,
- ``regime``: an evolvable taxonomy label.

Scenario rows may continue to carry historical task IDs.  Every accepted
identifier resolves to the same semantic task key and task bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from eeg_agent_bench.core.registry import Registry
from eeg_agent_bench.evaluators.base import BaseEvaluator
from eeg_agent_bench.tasks.base import BaseTask


@dataclass
class TaskBundle:
    """Bundle of task implementation and evaluators."""

    task: BaseTask
    evaluators: list[BaseEvaluator]


@dataclass(frozen=True)
class TaskMeta:
    """Stable identity, presentation metadata, and provenance for one task."""

    task_key: str
    display_id: str
    legacy_ids: tuple[str, ...]
    regime: str
    display_name_en: str
    datasets: tuple[str, ...]

    @property
    def task_id(self) -> str:
        """Backward-compatible access to the current presentation ID."""

        return self.display_id


REGIME_ORDER = (
    "knowledge_reasoning",
    "short_horizon_analysis",
    "long_horizon_analysis",
)

REGIME_NAMES = {
    "knowledge_reasoning": "Knowledge Reasoning",
    "short_horizon_analysis": "Short-Horizon Analysis",
    "long_horizon_analysis": "Long-Horizon Analysis",
}

TASK_META: dict[str, TaskMeta] = {
    "knowledge_qa": TaskMeta(
        task_key="knowledge_qa",
        display_id="T1",
        legacy_ids=("C6", "K0"),
        regime="knowledge_reasoning",
        display_name_en="EEG Knowledge QA",
        datasets=("MedMCQA-EEG-strict",),
    ),
    "artifact_identification": TaskMeta(
        task_key="artifact_identification",
        display_id="T2",
        legacy_ids=("C2", "C2-Artifact"),
        regime="short_horizon_analysis",
        display_name_en="Artifact Contamination Identification",
        datasets=("EEGdenoiseNet",),
    ),
    "epilepsy_screening": TaskMeta(
        task_key="epilepsy_screening",
        display_id="T3",
        legacy_ids=("C3", "C3-Routine"),
        regime="short_horizon_analysis",
        display_name_en="Normal-vs-Epileptic Screening",
        datasets=("Bonn",),
    ),
    "dementia_cohort_classification": TaskMeta(
        task_key="dementia_cohort_classification",
        display_id="T4",
        legacy_ids=("C4", "C3-Cohort"),
        regime="short_horizon_analysis",
        display_name_en="Dementia Cohort Classification",
        datasets=("ds004504",),
    ),
    "seizure_detection": TaskMeta(
        task_key="seizure_detection",
        display_id="T5",
        legacy_ids=("C1", "C1-Seizure"),
        regime="long_horizon_analysis",
        display_name_en="Seizure Event Detection",
        datasets=("CHB-MIT",),
    ),
    "sleep_staging": TaskMeta(
        task_key="sleep_staging",
        display_id="T6",
        legacy_ids=("C5", "C3-Sleep"),
        regime="long_horizon_analysis",
        display_name_en="Sleep Staging",
        datasets=("Sleep-EDFx",),
    ),
}

_IDENTIFIER_TO_KEY: dict[str, str] = {}
for _task_key, _meta in TASK_META.items():
    for _identifier in (_task_key, _meta.display_id, *_meta.legacy_ids):
        if _identifier in _IDENTIFIER_TO_KEY:
            raise RuntimeError(f"Duplicate task identifier: {_identifier}")
        _IDENTIFIER_TO_KEY[_identifier] = _task_key

TASK_ALIASES: dict[str, list[str]] = {
    task_key: [meta.display_id, *meta.legacy_ids]
    for task_key, meta in TASK_META.items()
}


def resolve_task_key(task_id: str) -> str:
    """Resolve a semantic key, display ID, or supported legacy ID."""

    return _IDENTIFIER_TO_KEY.get(task_id, task_id)


def canonical_task_id(task_id: str) -> str:
    """Backward-compatible alias for :func:`resolve_task_key`."""

    return resolve_task_key(task_id)


def display_task_id(task_id: str) -> str:
    """Return the current paper/leaderboard ID for any accepted identifier."""

    return TASK_META[resolve_task_key(task_id)].display_id


def get_task_meta(task_id: str) -> TaskMeta:
    """Return metadata for a semantic key, display ID, or legacy ID."""

    return TASK_META[resolve_task_key(task_id)]


def list_task_meta() -> list[TaskMeta]:
    """List tasks in paper order (T1 through T6)."""

    return sorted(TASK_META.values(), key=lambda meta: int(meta.display_id[1:]))


_registry: Registry[TaskBundle] = Registry("task")


def register_task(task_id: str, bundle: TaskBundle) -> None:
    if task_id not in _registry:
        _registry.register(task_id, bundle)


def get_task_bundle(task_id: str) -> TaskBundle:
    """Return the bundle for any accepted task identifier."""

    return _registry.get(task_id)


def list_task_ids() -> list[str]:
    return _registry.list_names()


def is_registered(task_id: str) -> bool:
    return task_id in _registry


def _register_defaults() -> None:
    from eeg_agent_bench.evaluators.artifact_classification import (
        ArtifactClassificationEvaluator,
    )
    from eeg_agent_bench.evaluators.classification import (
        ClassificationAccuracyEvaluator,
    )
    from eeg_agent_bench.evaluators.hypnogram_metrics import (
        HypnogramArchitectureErrorEvaluator,
        HypnogramCohenKappaEvaluator,
        HypnogramMacroF1Evaluator,
        HypnogramTransitionAccuracyEvaluator,
    )
    from eeg_agent_bench.evaluators.mcq_accuracy import MCQAccuracyEvaluator
    from eeg_agent_bench.evaluators.seizure_event import (
        SeizureDiceEvaluator,
        SeizureEventF1Evaluator,
        SeizureFalseAlarmEvaluator,
    )
    from eeg_agent_bench.tasks.t1_knowledge import T1KnowledgeQATask
    from eeg_agent_bench.tasks.t2_artifact import T2ArtifactTask
    from eeg_agent_bench.tasks.t3_epilepsy import T3EpilepsyTask
    from eeg_agent_bench.tasks.t4_dementia import T4DementiaTask
    from eeg_agent_bench.tasks.t5_seizure import T5SeizureTask
    from eeg_agent_bench.tasks.t6_sleep import T6SleepTask

    bundles: dict[str, TaskBundle] = {
        "knowledge_qa": TaskBundle(
            task=T1KnowledgeQATask(),
            evaluators=[MCQAccuracyEvaluator()],
        ),
        "artifact_identification": TaskBundle(
            task=T2ArtifactTask(),
            evaluators=[ArtifactClassificationEvaluator()],
        ),
        "epilepsy_screening": TaskBundle(
            task=T3EpilepsyTask(),
            evaluators=[ClassificationAccuracyEvaluator()],
        ),
        "dementia_cohort_classification": TaskBundle(
            task=T4DementiaTask(),
            evaluators=[ClassificationAccuracyEvaluator()],
        ),
        "seizure_detection": TaskBundle(
            task=T5SeizureTask(),
            evaluators=[
                SeizureDiceEvaluator(),
                SeizureFalseAlarmEvaluator(),
                SeizureEventF1Evaluator(),
            ],
        ),
        "sleep_staging": TaskBundle(
            task=T6SleepTask(),
            evaluators=[
                HypnogramMacroF1Evaluator(),
                HypnogramCohenKappaEvaluator(),
                HypnogramTransitionAccuracyEvaluator(),
                HypnogramArchitectureErrorEvaluator(),
            ],
        ),
    }
    for task_key, bundle in bundles.items():
        register_task(task_key, bundle)
        for alias in TASK_ALIASES[task_key]:
            register_task(alias, bundle)


_register_defaults()

__all__ = [
    "REGIME_NAMES",
    "REGIME_ORDER",
    "TASK_ALIASES",
    "TASK_META",
    "TaskBundle",
    "TaskMeta",
    "canonical_task_id",
    "display_task_id",
    "get_task_bundle",
    "get_task_meta",
    "is_registered",
    "list_task_ids",
    "list_task_meta",
    "register_task",
    "resolve_task_key",
]
