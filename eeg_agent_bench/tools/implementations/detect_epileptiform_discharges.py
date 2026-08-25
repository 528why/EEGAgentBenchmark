"""Tool: detect_transients — detect candidate transient peaks and measure morphology.

Pure signal-level peak detection + morphological measurement.
Returns *candidate* peaks ranked by amplitude — NOT confirmed clinical
events.  The model is responsible for interpreting whether the
morphology is consistent with epileptiform discharges, artifacts, or
normal physiological transients.

No clinical category labels, confidence scores, or diagnostic
interpretations are emitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class DetectTransientsTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="detect_transients",
            version="0.2.0",
            description=(
                "Detect candidate transient amplitude peaks in EEG channels "
                "and measure their morphology: peak time, half-width duration "
                "(ms), peak amplitude (µV), sharpness (µV/s), and polarity. "
                "Returns peak candidates ranked by amplitude with per-channel "
                "background statistics (RMS, peak-to-peak).  This is a raw "
                "signal-level peak detector — downstream interpretation of "
                "candidate morphology is the analyst's responsibility.\n"
                "\n"
                "## Output size\n"
                "Bounded by max_candidates_per_channel x n_channels (default "
                "20 x n_ch).  Returns: per-channel summary (~10 fields) + "
                "global top-5 candidates.  Even 64-ch with default cap stays "
                "under ~30 KB / ~8k tokens.  Lower min_amplitude_uv yields "
                "more candidates before the cap kicks in.\n"
                "\n"
                "## Applicability\n"
                "Works on any channel count incl. single-channel; the analysed "
                "window needs at least ~64 samples (zero-phase band-pass). "
                "Unknown channels or an empty/too-short window return a clear error."
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
                        "description": "Channels to scan. If omitted, uses all EEG channels.",
                    },
                    "tmin": {
                        "type": "number",
                        "description": "Start time in seconds.",
                    },
                    "tmax": {
                        "type": "number",
                        "description": "End time in seconds.",
                    },
                    "min_amplitude_uv": {
                        "type": "number",
                        "description": "Minimum peak amplitude in µV. Only candidates above this threshold are returned.",
                        "default": 70,
                    },
                    "max_candidates_per_channel": {
                        "type": "integer",
                        "description": "Maximum number of candidate peaks to return per channel, ranked by amplitude.",
                        "default": 20,
                    },
                },
                "required": ["record_id"],
            },
        )

    # Minimum inter-peak distance in seconds (70 ms), consistent with
    # the IFCN spike duration lower bound documented in Tool Reference.
    _PEAK_DISTANCE_SEC = 0.07
    # Bandpass filter band for isolating transients.
    _FILTER_BAND_HZ = (1.0, 70.0)

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, get_eeg_channels, get_data_array, _round,
            attach_response_meta,
            resolve_valid_channels, validate_time_window,
        )
        from scipy import signal as sig

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")
        min_amp = arguments.get("min_amplitude_uv", 70)
        # Accept both old and new parameter name for backward compat
        max_per_ch = arguments.get(
            "max_candidates_per_channel",
            arguments.get("max_events_per_channel", 20),
        )

        raw = load_raw(record_id, context)
        valid_chs, ch_err = resolve_valid_channels(raw, channels)
        if ch_err:
            return {"record_id": record_id, **ch_err}
        # Zero-phase band-pass (sosfiltfilt) needs the window to exceed the
        # filter padlen; require ≥64 samples to avoid a cryptic filtfilt error.
        win_err = validate_time_window(raw, tmin, tmax, min_samples=64)
        if win_err:
            return {"record_id": record_id, **win_err}
        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)

        offset = tmin if tmin is not None else 0.0
        min_amp_v = min_amp * 1e-6  # µV → V

        lo, hi_target = self._FILTER_BAND_HZ
        nyq = sfreq / 2
        hi = min(hi_target, nyq - 1)

        # Total candidates detected before top-K truncation (across channels).
        n_detected_before_topk = 0
        all_candidates: list[dict[str, Any]] = []
        # Per-channel background statistics (computed on filtered signal).
        background_stats: dict[str, dict[str, float]] = {}

        for i, ch in enumerate(valid_chs):
            x = data[i]

            if hi <= lo:
                continue
            sos = sig.butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
            x_filt = sig.sosfiltfilt(sos, x)

            # ── Background context (on filtered signal) ──────────
            bg_rms = float(np.sqrt(np.mean(x_filt ** 2))) * 1e6
            bg_p2p = float((np.max(x_filt) - np.min(x_filt))) * 1e6
            background_stats[ch] = {
                "background_rms_uv": _round(bg_rms, 1),
                "background_peak_to_peak_uv": _round(bg_p2p, 1),
            }

            # Derivative for sharpness
            dx = np.diff(x_filt) * sfreq  # V/s

            # Detect peaks (both positive and negative)
            dist_samples = max(1, int(sfreq * self._PEAK_DISTANCE_SEC))
            height = min_amp_v if min_amp_v > 0 else None
            peaks_pos, _ = sig.find_peaks(x_filt, height=height, distance=dist_samples)
            peaks_neg, _ = sig.find_peaks(-x_filt, height=height, distance=dist_samples)

            ch_candidates: list[dict[str, Any]] = []
            for pk, polarity in [(p, "positive") for p in peaks_pos] + [(p, "negative") for p in peaks_neg]:
                amp_uv = float(np.abs(x_filt[pk])) * 1e6

                if min_amp > 0 and amp_uv < min_amp:
                    continue

                time_sec = float(pk / sfreq) + offset

                # Half-width at half-max
                half_amp = np.abs(x_filt[pk]) / 2
                left = pk
                while left > 0 and np.abs(x_filt[left]) > half_amp:
                    left -= 1
                right = pk
                while right < len(x_filt) - 1 and np.abs(x_filt[right]) > half_amp:
                    right += 1
                width_ms = float((right - left) / sfreq * 1000)

                # Sharpness: max |derivative| near peak (±50ms)
                pk_start = max(0, pk - int(sfreq * self._PEAK_DISTANCE_SEC))
                pk_end = min(len(dx) - 1, pk + int(sfreq * self._PEAK_DISTANCE_SEC))
                sharpness = float(np.max(np.abs(dx[pk_start:pk_end]))) * 1e6 if pk_end > pk_start else 0

                ch_candidates.append({
                    "channel": ch,
                    "time_sec": _round(time_sec, 2),
                    "width_ms": _round(width_ms, 1),
                    "amplitude_uv": _round(amp_uv, 1),
                    "sharpness_uv_s": _round(sharpness, 1),
                    "polarity": polarity,
                })

            n_detected_before_topk += len(ch_candidates)

            # Keep top-K per channel by amplitude
            ch_candidates.sort(key=lambda e: e["amplitude_uv"], reverse=True)
            all_candidates.extend(ch_candidates[:max_per_ch])

        # ── Per-channel summary ───────────────────────────────────
        from collections import defaultdict
        ch_stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "amps": [], "widths": [], "sharpness": []}
        )
        for e in all_candidates:
            s = ch_stats[e["channel"]]
            s["count"] += 1
            s["amps"].append(e["amplitude_uv"])
            s["widths"].append(e["width_ms"])
            s["sharpness"].append(e["sharpness_uv_s"])

        channel_summary = []
        for ch, s in ch_stats.items():
            bg = background_stats.get(ch, {})
            bg_rms = bg.get("background_rms_uv", 0)
            amp_mean = float(np.mean(s["amps"]))
            amp_max = float(np.max(s["amps"]))
            channel_summary.append({
                "channel": ch,
                "n_candidates": s["count"],
                "amplitude_mean_uv": _round(amp_mean, 1),
                "amplitude_max_uv": _round(amp_max, 1),
                "width_mean_ms": _round(float(np.mean(s["widths"])), 1),
                "sharpness_mean_uv_s": _round(float(np.mean(s["sharpness"])), 1),
                # Background context for the analyst
                "background_rms_uv": bg.get("background_rms_uv", 0),
                "background_peak_to_peak_uv": bg.get("background_peak_to_peak_uv", 0),
                "amplitude_to_rms_ratio_mean": _round(amp_mean / bg_rms, 2) if bg_rms > 0 else None,
                "amplitude_to_rms_ratio_max": _round(amp_max / bg_rms, 2) if bg_rms > 0 else None,
            })

        # Global top-5 by amplitude
        all_candidates.sort(key=lambda e: e["amplitude_uv"], reverse=True)
        top_candidates = all_candidates[:5]

        # Note: the per-channel top-K cap is the truncation mechanism;
        # signal that to the LLM via _meta when we discarded candidates.
        _truncated = n_detected_before_topk > len(all_candidates)
        return attach_response_meta(
            {
                "record_id": record_id,
                "time_window": [tmin or 0, tmax],
                # ── Detection parameters (so the analyst knows the config) ──
                "min_amplitude_uv": min_amp,
                "peak_distance_ms": _round(self._PEAK_DISTANCE_SEC * 1000, 1),
                "filter_band_hz": list(self._FILTER_BAND_HZ),
                "max_candidates_per_channel": max_per_ch,
                "candidate_sort_key": "amplitude_descending",
                # ── Candidate counts ─────────────────────────────────────────
                "n_candidates_detected": n_detected_before_topk,
                "n_candidates_returned": len(all_candidates),
                # ── Results ──────────────────────────────────────────────────
                "channel_summary": channel_summary,
                "top_candidates": top_candidates,
            },
            n_rows=len(all_candidates),
            truncated=_truncated,
            truncated_reason=(
                f"per-channel top-{max_per_ch} cap "
                f"({n_detected_before_topk} detected -> {len(all_candidates)} kept)"
                if _truncated else None
            ),
        )
