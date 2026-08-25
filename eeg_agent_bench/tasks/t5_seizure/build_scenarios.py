"""Build T5-Seizure candidate scenarios from the CHB-MIT manifest.

The builder emits the frozen T5 seizure-detection scenario format.

- **1 EDF file = 1 scenario** (~1 hour long-record search).
- Both seizure and seizure-free files become scenarios — the seizure-free
  files are essential for the false-alarm (clinical-safety) metric.
- ``record_id`` is anonymised (``EEG_xxxx``) so the original ``chbXX_03``
  name cannot leak the answer; the real path travels in ``data_path``,
  which the prompt builder never shows the agent.  Tools resolve the
  anonymous id → path via the environment's ``record_paths`` map.
- Gold (seizure intervals) lives only in ``gold_private``.
"""

from __future__ import annotations

import logging
from typing import Any

from eeg_agent_bench.data.manifest import DatasetManifest, ManifestEntry
from eeg_agent_bench.tasks.t5_seizure.schema import T5_SEIZURE_OUTPUT_SCHEMA
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

# ── Tool list for T5-Seizure ──────────────────────────────────────────
# Fairness alignment: every task exposes the full analysis-tool catalog so
# that tool availability never leaks the expected approach; tool *selection*
# is part of what we evaluate.
T5_SEIZURE_TOOLS: list[str] = list(ALL_ANALYSIS_TOOLS)

_TASK_ID = "T5"
_TASK_NAME = "seizure_event_detection"
_DEFAULT_MAX_TURNS = 1000


def _access(max_turns: int = _DEFAULT_MAX_TURNS) -> AccessConfig:
    return AccessConfig(
        mode=AccessMode.METADATA_WITH_TOOLS,
        allowed_tools=list(T5_SEIZURE_TOOLS),
        max_turns=max_turns,
    )


def _evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(
        primary="seizure_event_f1",
        secondary=["seizure_false_alarms"],
    )


def _anon_id(_counter: dict[str, int] = {"n": 0}) -> str:
    _counter["n"] += 1
    return f"EEG_{_counter['n']:04d}"


def _public_metadata(entry: ManifestEntry) -> dict[str, Any]:
    """Metadata safe to expose — NEVER seizure_events / n_seizures."""
    meta = entry.metadata
    keys = {"sampling_rate", "n_channels", "duration_sec", "montage",
            "recording_type"}
    return {k: meta[k] for k in keys if k in meta and meta[k] is not None}


def _clinical_context(entry: ManifestEntry) -> str:
    meta = entry.metadata
    dur = meta.get("duration_sec")
    parts = [
        "continuous scalp EEG from a long-term epilepsy-monitoring session "
        "(pediatric epilepsy unit)",
        f"{int(meta.get('n_channels', 23))}-channel bipolar longitudinal montage",
        f"sampling rate {meta.get('sampling_rate', 256):.0f} Hz",
    ]
    if isinstance(dur, (int, float)) and dur > 0:
        parts.append(f"recording duration approximately {dur/60:.0f} minutes")
    return ", ".join(parts) + "."


def build_t5_seizure_chbmit_scenario(entry: ManifestEntry) -> Scenario:
    """Build one T5-Seizure scenario from a CHB-MIT manifest entry."""
    public_meta = _public_metadata(entry)
    anon_id = _anon_id()
    events = entry.metadata.get("seizure_events", []) or []

    return Scenario(
        scenario_id=f"c1_seizure_{entry.record_id}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="CHB-MIT",
        dataset_version=entry.dataset_version,
        records=[RecordRef(
            record_id=anon_id,
            role="target",
            data_path=entry.file_path,
            metadata=public_meta,
        )],
        input={
            "clinical_context": _clinical_context(entry),
            "question": (
                "Search the entire recording for electrographic seizures. "
                "Report each seizure as an onset/offset interval in seconds "
                "from the start of the recording. If there is no seizure, "
                "return an empty list."
            ),
            "metadata": public_meta,
        },
        access=_access(),
        expected_output_schema=T5_SEIZURE_OUTPUT_SCHEMA,
        gold_private={
            "label": entry.label,
            "has_seizure": entry.label == "seizure",
            "n_seizures": entry.metadata.get("n_seizures", len(events)),
            "seizure_events": [list(e) for e in events],
        },
        evaluator=_evaluator(),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


def build_t5_seizure_scenarios(manifest: DatasetManifest) -> list[Scenario]:
    """Build T5-Seizure scenarios for all eligible entries in *manifest*."""
    scenarios: list[Scenario] = []
    if manifest.dataset == "CHB-MIT":
        # Stable ordering: by subject then record id.
        for entry in sorted(manifest.entries.values(),
                            key=lambda e: (e.subject_id, e.record_id)):
            scenarios.append(build_t5_seizure_chbmit_scenario(entry))
    else:
        logger.warning(f"No T5-Seizure builder for dataset: {manifest.dataset}")
    return scenarios


__all__ = [
    "T5_SEIZURE_TOOLS",
    "build_t5_seizure_scenarios",
    "build_t5_seizure_chbmit_scenario",
]
