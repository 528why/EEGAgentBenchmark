"""Tool: get_recording_info — basic recording metadata."""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class GetRecordingInfoTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_recording_info",
            version="0.1.0",
            description=(
                "Return basic metadata for an EEG recording: sampling rate, "
                "duration, number of channels, reference scheme, montage, "
                "subject demographics, and data availability status. "
                "Applies to any recording; takes only ``record_id`` (no channel "
                "or time-window arguments). Output is a small bounded dict."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "EEG record identifier.",
                    },
                },
                "required": ["record_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "sampling_rate": {"type": "number"},
                    "duration_sec": {"type": "number"},
                    "n_channels": {"type": "integer"},
                    "n_eeg_channels": {"type": "integer"},
                    "channel_names": {"type": "array", "items": {"type": "string"}},
                    "reference": {"type": "string"},
                },
            },
        )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, get_eeg_channels, _round, attach_response_meta,
        )

        record_id = arguments["record_id"]
        raw = load_raw(record_id, context)
        eeg_chs = get_eeg_channels(raw)

        # Try to extract subject info from raw.info
        subject_info = raw.info.get("subject_info", {}) or {}

        return attach_response_meta(
            {
                "record_id": record_id,
                "sampling_rate": _round(raw.info["sfreq"], 1),
                "duration_sec": _round(raw.times[-1], 1),
                "n_channels": len(raw.ch_names),
                "n_eeg_channels": len(eeg_chs),
                "channel_names": raw.ch_names,
                "reference": raw.info.get("custom_ref_applied", "unknown"),
                "highpass_hz": _round(raw.info.get("highpass", 0), 2),
                "lowpass_hz": _round(raw.info.get("lowpass", 0), 2),
                "age": subject_info.get("age", None),
                "sex": subject_info.get("sex", None),
            },
            n_rows=1,
        )
