"""Build T6-Sleep candidate scenarios from dataset manifests.

The agent reads one whole-night Sleep-EDFx PSG and outputs a 30-second
hypnogram plus an architecture summary.

One scenario corresponds to one whole-night PSG. The full per-epoch label
sequence is moved into
``scenario.gold_private['epoch_labels_aasm']`` so the evaluator can
score it; nothing about the gold appears in the user/system prompt or
in any tool observation (because tools never receive the hypnogram
path — see access D3/D4 in the plan).
"""

from __future__ import annotations

import logging
from typing import Any

from eeg_agent_bench.data.manifest import DatasetManifest, ManifestEntry
from eeg_agent_bench.tasks.t6_sleep.schema import T6_SLEEP_OUTPUT_SCHEMA
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

AASM_LABELS: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
EPOCH_SEC = 30

# ── Tool list for T6-Sleep ────────────────────────────────────────────
# Fairness alignment: every task exposes the full analysis-tool catalog so
# tool availability never leaks the expected approach; tool *selection* is
# part of what we evaluate.
T6_SLEEP_TOOLS: list[str] = list(ALL_ANALYSIS_TOOLS)

_TASK_ID = "T6"
_TASK_NAME = "sleep_staging"

_DEFAULT_MAX_TURNS = 1000


def _sleep_access(max_turns: int = _DEFAULT_MAX_TURNS) -> AccessConfig:
    return AccessConfig(
        mode=AccessMode.METADATA_WITH_TOOLS,
        allowed_tools=list(T6_SLEEP_TOOLS),
        max_turns=max_turns,
    )


def _sleep_evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(
        primary="hypnogram_macro_f1",
        # Macro-F1 and Cohen's kappa are the headline metrics; transition
        # accuracy and architecture error are auxiliary diagnostics.
        secondary=[
            "hypnogram_cohen_kappa",
            "hypnogram_transition_accuracy",
            "hypnogram_architecture_error",
        ],
    )


def _clinical_context(entry: ManifestEntry) -> str:
    meta = entry.metadata
    age = meta.get("age")
    sex = meta.get("sex")
    source = meta.get("source", "")
    night = meta.get("night")
    n_epochs = meta.get("n_epochs_total", 0)
    duration_h = (entry.metadata.get("duration_sec", 0.0) or 0.0) / 3600.0

    parts: list[str] = []
    if age is not None and sex is not None:
        sex_word = "female" if sex == "F" else "male" if sex == "M" else "participant"
        parts.append(f"{age}-year-old {sex_word}")
    parts.append(
        "whole-night polysomnography (Sleep-EDFx, "
        + ("Sleep Cassette in-home" if source == "SC" else "Sleep Telemetry in-hospital")
        + (f", night {night}" if night else "")
        + ")"
    )
    if duration_h > 0:
        parts.append(f"recording duration ~{duration_h:.1f} hours")
    parts.append(
        f"{n_epochs} epochs of {EPOCH_SEC} s each (AASM 5-class scoring required)"
    )
    return ", ".join(parts) + "."


def _public_record_metadata(entry: ManifestEntry) -> dict[str, Any]:
    """Return the metadata dict that is safe to expose on the scenario.

    Crucially: NEVER include ``epoch_labels_aasm`` /
    ``epoch_label_counts_aasm`` (those are gold) or
    ``hypnogram_path``.  Architecture stats are *also* gold-derived,
    so they are excluded as well.
    """
    meta = entry.metadata
    safe_keys = {
        "source", "night", "age", "sex",
        "sampling_rate", "n_channels", "channels", "eeg_channels",
        "n_samples", "duration_sec", "epoch_sec", "n_epochs_total",
    }
    return {k: meta[k] for k in safe_keys if k in meta}


def _gold_private_for(entry: ManifestEntry) -> dict[str, Any]:
    """Compute the gold payload (per-epoch labels + architecture stats)."""
    epoch_labels: list[str] = entry.metadata.get("epoch_labels_aasm", [])
    n = len(epoch_labels)
    counts = entry.metadata.get("epoch_label_counts_aasm", {})
    if n > 0:
        rem_pct = counts.get("REM", 0) / n
        n3_pct = counts.get("N3", 0) / n
        sleep_efficiency = sum(counts.get(s, 0) for s in ("N1", "N2", "N3", "REM")) / n
    else:
        rem_pct = n3_pct = sleep_efficiency = 0.0
    return {
        "label": entry.label,
        "epoch_labels_aasm": list(epoch_labels),
        "epoch_label_counts_aasm": dict(counts),
        "n_epochs_total": n,
        "architecture": {
            "REM_pct": float(rem_pct),
            "N3_pct": float(n3_pct),
            "sleep_efficiency": float(sleep_efficiency),
        },
        "aasm_labels": list(AASM_LABELS),
    }


def _sleep_anon_id(record_id: str, _counter: dict[str, int] = {"n": 0}) -> str:
    """Sequential anonymous record ID hiding the real Sleep-EDF id + dataset."""
    _counter["n"] += 1
    return f"EEG_{_counter['n']:04d}"


def build_t6_sleep_sleep_edf_scenario(entry: ManifestEntry) -> Scenario:
    """Build a single T6-Sleep scenario from a Sleep-EDFx manifest entry."""
    record_metadata = _public_record_metadata(entry)

    n_epochs = record_metadata.get("n_epochs_total", 0)
    source = entry.metadata.get("source", "sc").lower() or "sc"

    return Scenario(
        scenario_id=f"c3_sleep_{source}_{entry.record_id}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="Sleep-EDFx",
        dataset_version=entry.dataset_version,
        records=[RecordRef(
            # Anonymous, label-free id; the real Sleep-EDF id (SC4001E0) and
            # the dataset identity are hidden from the agent.  tool_env maps
            # this id back to data_path at runtime via record_paths.
            record_id=_sleep_anon_id(entry.record_id),
            role="target",
            data_path=entry.file_path,
            metadata=record_metadata,
        )],
        input={
            "clinical_context": _clinical_context(entry),
            "question": (
                f"Score this whole-night PSG in {EPOCH_SEC}-second epochs "
                f"using AASM 5-class labels (W / N1 / N2 / N3 / REM).  Cover "
                f"all {n_epochs} epochs with a run-length-encoded hypnogram, "
                f"and report REM_pct, N3_pct, and sleep_efficiency for the "
                f"full recording."
            ),
            "metadata": record_metadata,
        },
        access=_sleep_access(),
        expected_output_schema=T6_SLEEP_OUTPUT_SCHEMA,
        gold_private=_gold_private_for(entry),
        evaluator=_sleep_evaluator(),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


def build_t6_sleep_scenarios(manifest: DatasetManifest) -> list[Scenario]:
    """Build T6-Sleep scenarios for all eligible entries in *manifest*."""
    scenarios: list[Scenario] = []
    dataset = manifest.dataset

    if dataset == "Sleep-EDFx":
        for entry in manifest.entries.values():
            if entry.metadata.get("n_epochs_total", 0) <= 0:
                logger.debug(
                    f"Skipping Sleep-EDFx entry without epochs: {entry.record_id}"
                )
                continue
            scenarios.append(build_t6_sleep_sleep_edf_scenario(entry))

    else:
        logger.warning(f"No T6-Sleep builder for dataset: {dataset}")

    return scenarios


__all__ = [
    "T6_SLEEP_TOOLS",
    "build_t6_sleep_scenarios",
    "build_t6_sleep_sleep_edf_scenario",
]
