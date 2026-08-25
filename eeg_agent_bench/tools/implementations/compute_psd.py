"""Tool: compute_psd — power spectral density, alpha peak, dominant freq."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputePSDTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_psd",
            version="0.1.0",
            description=(
                "Compute power spectral density (PSD) summary for specified "
                "EEG channels. Returns per-channel scalars: alpha peak "
                "(frequency + power), dominant frequency, total power, and "
                "5-band power dict (delta/theta/alpha/beta/gamma). "
                "Supports Welch and multitaper methods.\n"
                "\n"
                "## Output size\n"
                "Bounded: ~7 scalar fields x n_channels.  Even 64-ch dense "
                "montages stay <2 KB / <0.5k tokens.  Does NOT return the "
                "full PSD array; use band_powers_uv2 dict for band-level "
                "energy and call again per time window if temporal "
                "resolution is needed.\n"
                "\n"
                "## Applicability\n"
                "Works on any channel count incl. single-channel; needs a window "
                "of at least ~0.5 s for usable spectral resolution. Unknown "
                "channels or an empty/too-short window return a clear error."
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
                        "description": "Channels to compute PSD for. If omitted, uses all EEG channels.",
                    },
                    "tmin": {
                        "type": "number",
                        "description": "Start time in seconds. If omitted, uses recording start.",
                    },
                    "tmax": {
                        "type": "number",
                        "description": "End time in seconds. If omitted, uses recording end.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["welch", "multitaper"],
                        "description": "PSD estimation method.",
                        "default": "welch",
                    },
                    "fmin": {
                        "type": "number",
                        "description": "Minimum frequency in Hz.",
                        "default": 0.5,
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Maximum frequency in Hz.",
                        "default": 45,
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
        method = arguments.get("method", "welch")
        fmin = arguments.get("fmin", 0.5)
        fmax = arguments.get("fmax", 45)

        raw = load_raw(record_id, context)
        valid_chs, ch_err = resolve_valid_channels(raw, channels)
        if ch_err:
            return {"record_id": record_id, **ch_err}
        # Spectral estimation needs a non-trivial window (≥ ~0.5 s) so the
        # Welch grid has usable frequency resolution; reject empty/short.
        win_err = validate_time_window(
            raw, tmin, tmax, min_samples=max(8, int(raw.info["sfreq"] // 2))
        )
        if win_err:
            return {"record_id": record_id, **win_err}

        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)
        n_channels, n_samples = data.shape
        nperseg = min(int(sfreq * 2), n_samples)

        # Compute PSD per channel
        channel_results = []
        all_psd = []

        for i, ch in enumerate(valid_chs):
            if method == "multitaper":
                from scipy.signal.windows import dpss
                NW = 4
                n_tapers = 2 * NW - 1
                tapers = dpss(n_samples, NW, Kmax=n_tapers)
                psd_sum = np.zeros(nperseg // 2 + 1)
                for taper in tapers:
                    f, px = sig.welch(data[i] * taper[:n_samples], fs=sfreq, nperseg=nperseg)
                    psd_sum += px
                psd = psd_sum / n_tapers
                freqs = f
            else:
                freqs, psd = sig.welch(data[i], fs=sfreq, nperseg=nperseg)

            # Frequency mask
            mask = (freqs >= fmin) & (freqs <= fmax)
            freqs_out = freqs[mask]
            psd_out = psd[mask]

            # Alpha peak (8-13 Hz)
            alpha_mask = (freqs_out >= 8) & (freqs_out <= 13)
            alpha_peak_freq = None
            alpha_peak_power = None
            if alpha_mask.any() and psd_out[alpha_mask].max() > 0:
                alpha_idx = np.argmax(psd_out[alpha_mask])
                alpha_peak_freq = _round(freqs_out[alpha_mask][alpha_idx])
                alpha_peak_power = _round(float(psd_out[alpha_mask][alpha_idx]) * 1e12)  # V²/Hz→µV²/Hz

            # Dominant frequency
            dom_idx = np.argmax(psd_out)
            dom_freq = _round(freqs_out[dom_idx])

            # Band powers from PSD
            band_powers = {}
            total_power = integrate_power(psd_out, freqs_out)
            for band_name, (f_lo, f_hi) in BANDS.items():
                b_mask = (freqs_out >= f_lo) & (freqs_out <= f_hi)
                if b_mask.any():
                    band_powers[band_name] = _round(
                        integrate_power(psd_out[b_mask], freqs_out[b_mask]) * 1e12
                    )

            channel_results.append({
                "channel": ch,
                "alpha_peak_freq_hz": alpha_peak_freq,
                "alpha_peak_power_uv2hz": alpha_peak_power,
                "dominant_freq_hz": dom_freq,
                "total_power_uv2": _round(total_power * 1e12),
                "band_powers_uv2": band_powers,
            })
            all_psd.append(psd_out)

        # ── Global average PSD alpha peak ─────────────────────────
        mean_psd = np.mean(all_psd, axis=0) if all_psd else np.array([])
        global_alpha_peak = None
        if len(mean_psd) > 0:
            alpha_mask_g = (freqs_out >= 8) & (freqs_out <= 13)
            if alpha_mask_g.any():
                global_alpha_peak = _round(freqs_out[alpha_mask_g][np.argmax(mean_psd[alpha_mask_g])])

        return attach_response_meta(
            {
                "record_id": record_id,
                "method": method,
                "time_window": [tmin, tmax],
                "freq_range_hz": [fmin, fmax],
                "n_channels": len(valid_chs),
                "global_alpha_peak_hz": global_alpha_peak,
                "channels": channel_results,
            },
            n_rows=len(valid_chs),
        )
