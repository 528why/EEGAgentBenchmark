"""Context-budget management is a *tested* capability.

`compute_windowed_features` enforces a hard 8000-token output budget with a
PRE-FLIGHT check: an over-budget call is REJECTED with an actionable error and
nothing is computed (no silent truncation), so the agent can rescope.  A scoped
call within budget returns a complete result.

Plain assert-functions (the repo can run tests without pytest).  Skips
automatically if the audit data file is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

from eeg_agent_bench.tools.implementations import ALL_TOOLS
from eeg_agent_bench.tools.implementations.compute_windowed_features import (
    MAX_OBS_TOKENS,
)

ROOT = Path(__file__).resolve().parents[1]
_MC1 = ROOT / "workspace/raw_data/ds004504-1.0.8/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
_HAVE_DATA = _MC1.exists()

CTX = {
    "record_paths": {"MC1": str(_MC1)},
    "record_metadata": {"MC1": {"sampling_rate": 500.0}},
}
_TOOL = {t().spec.name: t() for t in ALL_TOOLS}["compute_windowed_features"]


def test_over_budget_call_is_rejected_not_truncated():
    """Whole-record fine-window call (huge output) must be rejected, not run."""
    if not _HAVE_DATA:
        return  # skip when audit data not present
    r = _TOOL.execute(
        {
            "record_id": "MC1",
            "features": ["theta_alpha_ratio", "slow_fast_ratio"],
            "window_sec": 1.0,
            "step_sec": 1.0,
        },
        CTX,
    )
    assert isinstance(r, dict)
    # Rejected with an actionable error, and NOTHING computed.
    assert r.get("error"), f"expected an error, got: {str(r)[:160]}"
    assert "channels" not in r, "rejected call must not return computed rows"
    assert r.get("budget_tokens") == MAX_OBS_TOKENS
    assert r.get("estimated_tokens", 0) > MAX_OBS_TOKENS
    flat = json.dumps(r, default=str).lower()
    assert "nothing was computed" in flat
    # Error must guide rescoping (channels / time range).
    assert "rescope" in flat or "tmin" in flat or "channel" in flat


def test_scoped_call_within_budget_succeeds():
    """A single-channel, short-span call fits the budget and returns rows."""
    if not _HAVE_DATA:
        return  # skip when audit data not present
    # First discover a valid channel name from the rejection payload's record.
    r = _TOOL.execute(
        {
            "record_id": "MC1",
            "features": ["theta_alpha_ratio"],
            "window_sec": 2.0,
            "step_sec": 2.0,
            "tmin": 0.0,
            "tmax": 20.0,
        },
        CTX,
    )
    assert isinstance(r, dict)
    assert not r.get("error"), f"scoped call should succeed, got: {str(r)[:160]}"
    assert r.get("channels"), "scoped call must return computed channel rows"
    meta = r.get("_meta", {})
    # Within budget, and truthfully reported as complete (not truncated).
    assert meta.get("truncated") is False
    assert meta.get("approx_response_tokens", 0) <= MAX_OBS_TOKENS
