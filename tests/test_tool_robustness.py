"""Tools must fail loudly on degenerate input.

A degenerate call must return a dict with ``error`` (or explicit
``applicable: False``) — never raise, never return a silent-empty "ok".

Written as plain assert-functions (the repo runs tests without pytest).
Skips automatically if the audit data files are not present.
"""
from __future__ import annotations

import json
from pathlib import Path

from eeg_agent_bench.tools.implementations import ALL_TOOLS

ROOT = Path(__file__).resolve().parents[1]
_SC1 = ROOT / "workspace/processed/eegdenoisenet/signals/edn_clean_0000.npy"
_MC1 = ROOT / "workspace/raw_data/ds004504-1.0.8/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
_HAVE_DATA = _SC1.exists() and _MC1.exists()

CTX = {
    "record_paths": {"SC1": str(_SC1), "MC1": str(_MC1)},
    "record_metadata": {
        "SC1": {"sampling_rate": 256.0, "channel_names": ["EEG"], "n_samples": 512, "duration_sec": 2.0},
        "MC1": {"sampling_rate": 500.0},
    },
}
_TOOLS = {t().spec.name: t() for t in ALL_TOOLS}

_DEGENERATE = [
    ("compute_psd", {"tmin": 1.5, "tmax": 0.5}, "SC1"),
    ("compute_temporal_features", {"tmin": 1.5, "tmax": 0.5}, "SC1"),
    ("detect_transients", {"tmin": 1.5, "tmax": 0.5}, "SC1"),
    ("compute_psd", {"tmin": 100, "tmax": 200}, "SC1"),
    ("detect_transients", {"tmin": 100, "tmax": 200}, "SC1"),
    ("compute_psd", {"channels": ["NOPE"]}, "SC1"),
    ("compute_band_power", {"channels": ["NOPE"]}, "SC1"),
    ("compute_temporal_features", {"channels": ["NOPE"]}, "SC1"),
    ("compute_noise_metrics", {"channels": ["NOPE"]}, "SC1"),
    ("compute_asymmetry", {}, "SC1"),            # 1 channel -> not applicable
    ("compute_channel_correlation", {}, "SC1"),  # 1 channel -> error
]


def _failed_loud(r) -> bool:
    if not isinstance(r, dict):
        return False
    if r.get("status") == "error" or r.get("error"):
        return True
    flat = json.dumps(r, default=str).lower()
    return r.get("applicable") is False or "not applicable" in flat


def test_degenerate_inputs_fail_loud():
    if not _HAVE_DATA:
        return  # skip when audit data not present
    for tool, args, rid in _DEGENERATE:
        r = _TOOLS[tool].execute({"record_id": rid, **args}, CTX)  # must not raise
        assert _failed_loud(r), f"{tool} {args} did not fail loudly: {str(r)[:160]}"


def test_valid_call_still_ok():
    if not _HAVE_DATA:
        return
    r = _TOOLS["compute_psd"].execute({"record_id": "SC1"}, CTX)
    assert isinstance(r, dict) and not r.get("error")
