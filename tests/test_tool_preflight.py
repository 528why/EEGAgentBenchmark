"""Runtime preflight tests for EEG tool evaluation."""

from __future__ import annotations

from unittest.mock import patch

from eeg_agent_bench.tools.preflight import validate_tool_runtime
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import AccessConfig, Scenario


def _scenario(allowed_tools):
    return Scenario(
        scenario_id="preflight",
        task_id="C2",
        task_name="artifact_contamination",
        dataset="test",
        access=AccessConfig(allowed_tools=list(allowed_tools)),
    )


def test_preflight_rejects_unregistered_allowed_tool():
    try:
        validate_tool_runtime([_scenario(["missing_tool"])], ToolRegistry())
    except RuntimeError as exc:
        assert "unregistered tools" in str(exc)
    else:
        raise AssertionError("preflight should reject missing tool implementations")


def test_preflight_rejects_missing_runtime_dependency():
    registry = ToolRegistry()
    registry._specs["compute_psd"] = object()
    registry._registry._items["compute_psd"] = object()

    def import_module(name):
        if name == "mne":
            raise ImportError("missing")
        return object()

    with patch("eeg_agent_bench.tools.preflight.importlib.import_module", import_module):
        try:
            validate_tool_runtime([_scenario(["compute_psd"])], registry)
        except RuntimeError as exc:
            assert "mne" in str(exc)
        else:
            raise AssertionError("preflight should reject missing MNE")
