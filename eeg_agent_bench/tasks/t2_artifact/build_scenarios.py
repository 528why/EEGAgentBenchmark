"""Build T2-Artifact candidate scenarios from the EEGdenoiseNet manifest.

Reads ``workspace/candidate_manifests/eegdenoisenet_build.jsonl`` (produced by
``scripts/prepare_eegdenoisenet.py``) and emits one
:class:`Scenario` per synthesized epoch.

Data-leak contract:
- ``input`` carries only neutral epoch parameters + label space.
- ``data_path`` and ``record_id`` are artifact-neutral (the source/snr are
  NOT encoded in the id shown to the model — only an opaque id is rendered;
  see ``_anon_id``).
- ``gold_private`` carries label / artifact_source / snr_db.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from eeg_agent_bench.tasks.t2_artifact.schema import (
    T2_ARTIFACT_LABELS,
    T2_ARTIFACT_OUTPUT_SCHEMA,
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

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "workspace/candidate_manifests/eegdenoisenet_build.jsonl"

_TASK_ID = "T2"
_TASK_NAME = "artifact_contamination"

# Fairness alignment: expose the full analysis-tool catalog (same for every
# task).  Tools that need ≥2 channels (asymmetry / correlation) simply report
# "not applicable" on this single-channel epoch —
# recognising that is part of the test, not a service we pre-render.
# recognising that is part of the test, not a service we pre-render.
T2_ARTIFACT_TOOLS: list[str] = list(ALL_ANALYSIS_TOOLS)


def _anon_id(record_id: str, _counter: dict[str, int] = {"n": 0}) -> str:
    """Opaque sequential id so the model never sees source/snr in the id."""
    _counter["n"] += 1
    return f"EPOCH_{_counter['n']:05d}"


def _access(max_turns: int = 1000) -> AccessConfig:
    return AccessConfig(
        mode=AccessMode.METADATA_WITH_TOOLS,
        allowed_tools=list(T2_ARTIFACT_TOOLS),
        max_turns=max_turns,
    )


def build_t2_artifact_scenario(row: dict[str, Any]) -> Scenario:
    rid = row["record_id"]
    anon = _anon_id(rid)
    # Repo-root-relative path; tools resolve it
    # against the CWD, which is the repo root when ``eeg-bench`` runs.
    signal_path = row["signal_path"]
    sfreq = row.get("sampling_rate", 256.0)
    n_samples = row.get("n_samples", 512)
    duration = row.get("duration_sec", round(n_samples / sfreq, 4))

    public_meta = {
        "sampling_rate": sfreq,
        "channels": 1,
        "channel_names": ["EEG"],
        "n_samples": n_samples,
        "duration_sec": duration,
    }

    return Scenario(
        scenario_id=f"c2_artifact:{rid}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="EEGdenoiseNet",
        dataset_version="1.0",
        records=[RecordRef(
            record_id=anon,
            role="target",
            data_path=signal_path,
            metadata=public_meta,
        )],
        input={
            "clinical_context": (
                "A short single-channel EEG epoch "
                f"(~{duration:g}s at {sfreq:g} Hz). "
                "Assess its signal quality."
            ),
            "question": (
                "Is this EEG epoch clean, contaminated by ocular (EOG) "
                "artifact, or contaminated by muscle (EMG) artifact? "
                "Use the measurement tools, then give your decision."
            ),
            "label_space": list(T2_ARTIFACT_LABELS),
        },
        access=_access(),
        expected_output_schema=T2_ARTIFACT_OUTPUT_SCHEMA,
        gold_private={
            "label": row["label"],
            "artifact_source": row.get("artifact_source"),
            "snr_db": row.get("snr_db"),
        },
        evaluator=EvaluatorConfig(primary="artifact_correct", secondary=[]),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_t2_artifact_scenarios(
    rows: Iterable[dict[str, Any]],
    *,
    subset: str = "candidate",
) -> list[Scenario]:
    """Build scenarios for ``candidate`` (all), ``frozen`` or ``smoke``."""
    out: list[Scenario] = []
    for row in rows:
        if subset == "frozen" and not row.get("in_frozen"):
            continue
        if subset == "smoke" and not row.get("in_smoke"):
            continue
        out.append(build_t2_artifact_scenario(row))
    return out


__all__ = [
    "T2_ARTIFACT_TOOLS",
    "DEFAULT_MANIFEST",
    "build_t2_artifact_scenario",
    "build_t2_artifact_scenarios",
    "load_manifest",
]
