"""Tool: compute_noise_metrics — per-channel noise and artifact indicators."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputeNoiseMetricsTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_noise_metrics",
            version="0.1.0",
            description=(
                "Compute noise-related metrics for EEG channels: RMS amplitude (µV), "
                "flat-line ratio, saturation ratio, line-noise ratio (power at mains "
                "frequency vs broadband), and high-frequency noise ratio (>40 Hz vs "
                "broadband). "
                "Applies to any channel count incl. single-channel; aggregates over "
                "the full recording (no time-window arguments). Requesting channels "
                "that do not exist returns a clear error. Output is bounded "
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
                        "description": "Channels to assess. If omitted, uses all EEG channels.",
                    },
                    "line_freq": {
                        "type": "number",
                        "description": "Power line frequency in Hz (50 or 60).",
                        "default": 50,
                    },
                },
                "required": ["record_id"],
            },
        )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, get_eeg_channels, get_data_array, _round,
            attach_response_meta, resolve_valid_channels,
        )
        from scipy import signal as sig

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        line_freq = arguments.get("line_freq", 50)
        raw = load_raw(record_id, context)
        eeg_chs, ch_err = resolve_valid_channels(raw, channels)
        if ch_err:
            return {"record_id": record_id, **ch_err}
        data, sfreq = get_data_array(raw, channels=eeg_chs)

        n_channels, n_samples = data.shape
        results = []

        for i, ch in enumerate(eeg_chs):
            x = data[i]

            # Flat-line detection: ratio of near-zero derivative
            dx = np.diff(x)
            flat_ratio = float(np.mean(np.abs(dx) < 1e-7))

            # Saturation: ratio of samples at min/max
            sat_ratio = float(np.mean((x == x.max()) | (x == x.min())) if x.max() != x.min() else 1.0)

            # RMS amplitude (µV)
            rms = float(np.sqrt(np.mean(x ** 2))) * 1e6  # V→µV

            # Line noise: PSD at line_freq ± 2 Hz vs broadband
            freqs, pxx = sig.welch(x, fs=sfreq, nperseg=min(int(sfreq * 2), n_samples))
            mask_line = (freqs >= line_freq - 2) & (freqs <= line_freq + 2)
            mask_broad = (freqs >= 1) & (freqs <= sfreq / 2 - 1)
            line_power = float(np.mean(pxx[mask_line])) if mask_line.any() else 0
            broad_power = float(np.mean(pxx[mask_broad])) if mask_broad.any() else 1e-20
            line_noise_ratio = line_power / broad_power if broad_power > 0 else 0

            # High-freq noise: power > 40 Hz / total
            mask_hf = freqs > 40
            hf_ratio = float(np.sum(pxx[mask_hf]) / (np.sum(pxx[mask_broad]) + 1e-20))

            results.append({
                "channel": ch,
                "rms_uv": _round(rms, 1),
                "flat_line_ratio": _round(flat_ratio, 4),
                "saturation_ratio": _round(sat_ratio, 4),
                "line_noise_ratio": _round(line_noise_ratio, 2),
                "hf_noise_ratio": _round(hf_ratio, 3),
            })

        return attach_response_meta(
            {
                "record_id": record_id,
                "line_freq_hz": line_freq,
                "n_channels": n_channels,
                "channels": results,
            },
            n_rows=n_channels,
        )
