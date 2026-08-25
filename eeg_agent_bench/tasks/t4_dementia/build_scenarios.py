"""Build T4-Dementia candidate scenarios from dataset manifests.

Supported dataset: ds004504, a 19-channel resting-state EEG cohort with
labels AD / FTD / HC. This is explicitly non-diagnostic cohort discrimination.
"""

from __future__ import annotations

import logging
from typing import Any

from eeg_agent_bench.data.manifest import DatasetManifest, ManifestEntry
from eeg_agent_bench.tasks.t4_dementia.schema import (
    T4_DEMENTIA_DS004504_OUTPUT_SCHEMA,
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

# ── Tool list for T4-Dementia ───────────────────────────────────────────
# Fairness alignment: every task exposes the full analysis-tool catalog so
# tool availability never leaks the expected approach; tool *selection* is
# part of what we evaluate.
T4_DEMENTIA_TOOLS: list[str] = list(ALL_ANALYSIS_TOOLS)

_TASK_ID = "T4"
_TASK_NAME = "cohort_discrimination"


def _cohort_access(max_turns: int = 1000) -> AccessConfig:
    return AccessConfig(
        mode=AccessMode.METADATA_WITH_TOOLS,
        allowed_tools=list(T4_DEMENTIA_TOOLS),
        max_turns=max_turns,
    )


def _cohort_evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(
        primary="classification_correct",
        secondary=[],
    )


# ── ds004504 scenario builder ─────────────────────────────────────────


def _ds004504_clinical_context(entry: ManifestEntry) -> str:
    meta = entry.metadata
    age = meta.get("age", "unknown")
    gender_map = {"M": "male", "F": "female"}
    gender = gender_map.get(meta.get("gender", ""), "unknown sex")

    # NOTE: MMSE is deliberately NOT included.  It is a cognitive-impairment
    # score that acts as a near-label shortcut for the AD/FTD/HC cohort and
    # would let the model bypass EEG analysis (see help-level alignment).
    parts = [f"{age}-year-old {gender} participant"]
    parts.append("eyes-closed resting-state EEG recording")
    parts.append(
        f"19-channel 10-20 system, sampling rate "
        f"{meta.get('sampling_rate', 500)} Hz"
    )
    parts.append(
        f"recording duration approximately "
        f"{meta.get('duration_sec', 600):.0f} seconds"
    )
    return ", ".join(parts) + "."


def _cohort_anon_id(record_id: str, _counter: dict[str, int] = {"n": 0}) -> str:
    """Sequential anonymous record ID hiding the real BIDS id + dataset identity."""
    _counter["n"] += 1
    return f"EEG_{_counter['n']:04d}"


def build_t4_dementia_ds004504_scenario(entry: ManifestEntry) -> Scenario:
    """Build a single T4-Dementia scenario from a ds004504 manifest entry."""
    clinical_context = _ds004504_clinical_context(entry)

    return Scenario(
        scenario_id=f"c3_cohort_ds004504_{entry.record_id}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="ds004504",
        dataset_version=entry.dataset_version,
        records=[RecordRef(
            # Anonymous, label-free id; the real BIDS id (sub-001) and the
            # dataset identity are hidden from the agent.  tool_env maps this
            # id back to data_path at runtime via record_paths.
            record_id=_cohort_anon_id(entry.record_id),
            role="target",
            data_path=entry.file_path,
            metadata=entry.metadata,
        )],
        subjects=[],
        input={
            "clinical_context": clinical_context,
            "question": (
                "Based on this eyes-closed resting-state EEG and the "
                "research context, identify the participant's cohort: "
                "Alzheimer's disease (AD), frontotemporal dementia (FTD), "
                "or healthy control (HC).  This is a cohort label, not a "
                "diagnosis.  List the supporting EEG findings."
            ),
            "metadata": {
                "age": entry.metadata.get("age"),
                "gender": entry.metadata.get("gender"),
                # mmse intentionally omitted (near-label shortcut).
                "channels": entry.metadata.get("channels"),
                "channel_names": entry.metadata.get("channel_names", []),
                "sampling_rate": entry.metadata.get("sampling_rate"),
                "duration_sec": entry.metadata.get("duration_sec"),
                "montage": entry.metadata.get("montage"),
                "reference": entry.metadata.get("reference"),
            },
        },
        access=_cohort_access(),
        expected_output_schema=T4_DEMENTIA_DS004504_OUTPUT_SCHEMA,
        gold_private={
            "label": entry.label,
            "finding_tags": [],
        },
        evaluator=_cohort_evaluator(),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


# ── Unified dispatcher ────────────────────────────────────────────────


def build_t4_dementia_scenarios(manifest: DatasetManifest) -> list[Scenario]:
    """Build T4-Dementia scenarios for all eligible entries in a manifest."""
    scenarios: list[Scenario] = []
    dataset = manifest.dataset

    if dataset == "ds004504":
        for entry in manifest.entries.values():
            if entry.label in ("AD", "FTD", "HC"):
                scenarios.append(build_t4_dementia_ds004504_scenario(entry))
            else:
                logger.debug(
                    f"Skipping ds004504 entry with label '{entry.label}': "
                    f"{entry.record_id}"
                )

    else:
        logger.warning(f"No T4-Dementia builder for dataset: {dataset}")

    return scenarios


__all__ = [
    "T4_DEMENTIA_TOOLS",
    "build_t4_dementia_scenarios",
    "build_t4_dementia_ds004504_scenario",
]
