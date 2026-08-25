"""Tool: compute_band_power — absolute/relative band power and ratios."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputeBandPowerTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_band_power",
            version="0.1.0",
            description=(
                "Compute absolute power (µV²), relative power, and standard ratios "
                "for each frequency band (delta/theta/alpha/beta/gamma) per channel. "
                "Ratios include theta/alpha, delta/alpha, and slow/fast "
                "((delta+theta)/(alpha+beta)).\n"
                "\n"
                "## Output size\n"
                "Bounded: per-channel dict with 5 absolute + 5 relative + 3 "
                "ratio scalars; one row per channel.  Even 64-ch montages "
                "stay <3 KB / <1k tokens.  Aggregates over the whole "
                "[tmin, tmax] window — call separately per epoch if you "
                "need temporal resolution (or use compute_windowed_features).\n"
                "\n"
                "## Applicability\n"
                "Works on any channel count incl. single-channel; needs a window "
                "of at least ~0.5 s. Unknown channels or an empty/too-short "
                "window return a clear error."
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
            load_raw, get_eeg_channels, get_data_array, _round, BANDS,
            integrate_power, attach_response_meta,
            resolve_valid_channels, validate_time_window,
        )
        from scipy import signal as sig

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")

        raw = load_raw(record_id, context)
        valid_chs, ch_err = resolve_valid_channels(raw, channels)
        if ch_err:
            return {"record_id": record_id, **ch_err}
        win_err = validate_time_window(
            raw, tmin, tmax, min_samples=max(8, int(raw.info["sfreq"] // 2))
        )
        if win_err:
            return {"record_id": record_id, **win_err}
        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)

        n_channels, n_samples = data.shape
        nperseg = min(int(sfreq * 2), n_samples)

        channel_results = []
        all_band_abs: dict[str, list[float]] = {b: [] for b in BANDS}

        for i, ch in enumerate(valid_chs):
            freqs, psd = sig.welch(data[i], fs=sfreq, nperseg=nperseg)

            # Compute absolute power per band (µV²)
            abs_power: dict[str, float] = {}
            for band, (flo, fhi) in BANDS.items():
                mask = (freqs >= flo) & (freqs <= fhi)
                if mask.any():
                    abs_power[band] = integrate_power(psd[mask], freqs[mask]) * 1e12
                else:
                    abs_power[band] = 0.0
                all_band_abs[band].append(abs_power[band])

            total = sum(abs_power.values())
            total = max(total, 1e-20)

            # Relative power
            rel_power = {b: _round(v / total, 4) for b, v in abs_power.items()}

            # Ratios
            alpha_p = max(abs_power.get("alpha", 1e-20), 1e-20)
            theta_p = abs_power.get("theta", 0)
            delta_p = abs_power.get("delta", 0)
            beta_p = abs_power.get("beta", 0)

            ratios = {
                "theta_alpha_ratio": _round(theta_p / alpha_p),
                "delta_alpha_ratio": _round(delta_p / alpha_p),
                "slow_fast_ratio": _round(
                    (delta_p + theta_p) / max(alpha_p + beta_p, 1e-20)
                ),
            }

            channel_results.append({
                "channel": ch,
                "absolute_power_uv2": {b: _round(v) for b, v in abs_power.items()},
                "relative_power": rel_power,
                "ratios": ratios,
            })

        return attach_response_meta(
            {
                "record_id": record_id,
                "time_window": [tmin, tmax],
                "n_channels": len(valid_chs),
                "channels": channel_results,
            },
            n_rows=len(valid_chs),
        )
