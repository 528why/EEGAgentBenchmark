"""Tool: compute_temporal_features — statistical and Hjorth features."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputeTemporalFeaturesTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_temporal_features",
            version="0.1.0",
            description=(
                "Compute time-domain statistical features for EEG channels: "
                "mean, std, RMS (µV), peak-to-peak amplitude, skewness, kurtosis, "
                "Hjorth parameters (activity/mobility/complexity), line length "
                "(µV/s), and zero-crossing rate (Hz). "
                "Applies to any channel count incl. single-channel; aggregates over "
                "[tmin, tmax] (whole record if omitted). Unknown channels or an "
                "empty/too-short window return a clear error. Output is bounded "
                "(one row per channel)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "EEG record identifier.",
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Channels to analyze. If omitted, uses all EEG channels.",
                    },
                    "tmin": {
                        "type": "number",
                        "description": "Start time in seconds.",
                    },
                    "tmax": {
                        "type": "number",
                        "description": "End time in seconds.",
                    },
                },
                "required": ["record_id"],
            },
        )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, get_eeg_channels, get_data_array, _round,
            attach_response_meta,
            resolve_valid_channels, validate_time_window,
        )
        from scipy.stats import skew, kurtosis

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")

        raw = load_raw(record_id, context)
        valid_chs, ch_err = resolve_valid_channels(raw, channels)
        if ch_err:
            return {"record_id": record_id, **ch_err}
        # Hjorth/diff features need ≥3 samples; reject empty/degenerate windows.
        win_err = validate_time_window(raw, tmin, tmax, min_samples=3)
        if win_err:
            return {"record_id": record_id, **win_err}
        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)

        results = []
        for i, ch in enumerate(valid_chs):
            x = data[i]
            x_uv = x * 1e6  # V → µV for human-readable amplitudes

            # Basic stats
            mean_uv = float(np.mean(x_uv))
            std_uv = float(np.std(x_uv))
            rms_uv = float(np.sqrt(np.mean(x_uv ** 2)))
            ptp_uv = float(np.ptp(x_uv))

            # Higher-order stats
            skewness = float(skew(x_uv))
            kurt = float(kurtosis(x_uv))

            # Hjorth parameters (on raw-scale data)
            dx = np.diff(x)
            ddx = np.diff(dx)
            activity = float(np.var(x))
            mobility = float(np.sqrt(np.var(dx) / max(activity, 1e-30)))
            complexity = float(
                np.sqrt(np.var(ddx) / max(np.var(dx), 1e-30)) / max(mobility, 1e-30)
            )

            # Line length (sum of abs differences, normalised by duration)
            duration = len(x) / sfreq
            line_length = float(np.sum(np.abs(dx)) * 1e6 / max(duration, 1e-6))  # µV/s

            # Zero-crossing rate
            zc = float(np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / max(duration, 1e-6))

            results.append({
                "channel": ch,
                "mean_uv": _round(mean_uv, 2),
                "std_uv": _round(std_uv, 2),
                "rms_uv": _round(rms_uv, 2),
                "peak_to_peak_uv": _round(ptp_uv, 1),
                "skewness": _round(skewness, 3),
                "kurtosis": _round(kurt, 3),
                "hjorth_activity": _round(activity * 1e12, 4),  # µV²
                "hjorth_mobility": _round(mobility, 4),
                "hjorth_complexity": _round(complexity, 4),
                "line_length_uv_s": _round(line_length, 1),
                "zero_crossing_rate_hz": _round(zc, 2),
            })

        return attach_response_meta(
            {
                "record_id": record_id,
                "time_window": [tmin, tmax],
                "n_channels": len(valid_chs),
                "channels": results,
            },
            n_rows=len(valid_chs),
        )
