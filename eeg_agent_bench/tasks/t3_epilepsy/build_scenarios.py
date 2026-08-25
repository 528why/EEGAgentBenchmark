"""Build normal-vs-epileptic screening scenarios from the Bonn manifest.

Supported dataset: Bonn (sets Z and O are normal; set S is epileptic).
"""

from __future__ import annotations

import logging
from typing import Any

from eeg_agent_bench.data.manifest import DatasetManifest, ManifestEntry
from eeg_agent_bench.tasks.t3_epilepsy.schema import (
    T3_EPILEPSY_BONN_OUTPUT_SCHEMA,
)
from eeg_agent_bench.tools.catalog import ALL_ANALYSIS_TOOLS
from eeg_agent_bench.types import (
    AccessConfig,
    AccessMode,
    EvaluatorConfig,
    RecordRef,
    ReleaseConfig,
    Scenario,
)

logger = logging.getLogger(__name__)

# ── Tool list for T3-Epilepsy ──────────────────────────────────────────
# Fairness alignment: every task exposes the full analysis-tool catalog so
# tool availability never leaks the expected approach; tool *selection* is
# part of what we evaluate.
T3_EPILEPSY_TOOLS: list[str] = list(ALL_ANALYSIS_TOOLS)

_TASK_ID = "T3"
_TASK_NAME = "routine_impression"


def _routine_access(max_turns: int = 1000) -> AccessConfig:
    return AccessConfig(
        mode=AccessMode.METADATA_WITH_TOOLS,
        allowed_tools=list(T3_EPILEPSY_TOOLS),
        max_turns=max_turns,
    )


def _routine_evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(primary="classification_correct", secondary=[])


# ── Bonn scenario builder ─────────────────────────────────────────────


def _bonn_clinical_context(entry: ManifestEntry) -> str:
    meta = entry.metadata
    # NOTE: recording_type (surface/intracranial) is deliberately NOT mentioned.
    # In Bonn it is perfectly colinear with the label, so naming it here would
    # be a 100% label shortcut (surface==normal, intracranial==epileptic).
    parts = [
        "single-channel EEG recording",
        f"sampling rate {meta.get('sampling_rate', 173.61):.2f} Hz",
        f"duration approximately {meta.get('duration_sec', 23.59):.1f} seconds",
        f"{meta.get('n_samples', 4096)} samples",
    ]
    return ", ".join(parts) + "."


def _bonn_anon_id(record_id: str, _counter: dict[str, int] = {"n": 0}) -> str:
    """Sequential anonymous record ID to hide the Bonn set letter (Z/O/S)."""
    _counter["n"] += 1
    return f"EEG_{_counter['n']:04d}"


def build_t3_epilepsy_bonn_scenario(entry: ManifestEntry) -> Scenario:
    """Build a single Bonn screening scenario."""
    clinical_context = _bonn_clinical_context(entry)
    label_binary = entry.metadata.get("label_binary", entry.label)
    anon_id = _bonn_anon_id(entry.record_id)

    return Scenario(
        scenario_id=f"c3_routine_bonn_{entry.record_id}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="Bonn",
        dataset_version=entry.dataset_version,
        records=[RecordRef(
            record_id=anon_id,
            role="target",
            data_path=entry.file_path,
            metadata=entry.metadata,
        )],
        input={
            "clinical_context": clinical_context,
            "question": (
                "Classify this EEG recording as ``normal`` or ``epileptic`` "
                "and list the supporting findings."
            ),
            "metadata": {
                "recording_type": entry.metadata.get("recording_type"),
                "channels": entry.metadata.get("channels"),
                "sampling_rate": entry.metadata.get("sampling_rate"),
                "n_samples": entry.metadata.get("n_samples"),
                "duration_sec": entry.metadata.get("duration_sec"),
            },
        },
        access=_routine_access(),
        expected_output_schema=T3_EPILEPSY_BONN_OUTPUT_SCHEMA,
        gold_private={
            "label": label_binary,
            "label_binary": label_binary,
            "finding_tags": [],
        },
        evaluator=_routine_evaluator(),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


# ── Unified dispatcher ────────────────────────────────────────────────


def build_t3_epilepsy_scenarios(manifest: DatasetManifest) -> list[Scenario]:
    """Build T3-Epilepsy scenarios for all eligible entries in a manifest."""
    scenarios: list[Scenario] = []
    dataset = manifest.dataset

    if dataset == "Bonn":
        for entry in manifest.entries.values():
            # Only sets Z, O (normal) and S (epileptic) make sense for the
            # binary task.  Sets N, F (interictal) are out-of-scope for
            # this binary schema and are skipped.
            set_letter = entry.metadata.get("set_letter", "")
            if set_letter in ("Z", "O", "S"):
                scenarios.append(build_t3_epilepsy_bonn_scenario(entry))
            else:
                logger.debug(
                    f"Skipping Bonn set '{set_letter}' (interictal): {entry.record_id}"
                )

    else:
        logger.warning(f"No T3-Epilepsy builder for dataset: {dataset}")

    return scenarios


__all__ = [
    "T3_EPILEPSY_TOOLS",
    "build_t3_epilepsy_scenarios",
    "build_t3_epilepsy_bonn_scenario",
]
