"""Tool: compute_windowed_features — sliding-window batch feature extraction.

Returns per-window time-series of key features for specified channels.
Essential for artifact temporal localisation (T3) and sleep staging (T5)
where the agent needs to see how features vary over time.

Design: purely non-parametric.  Each window is an independent statistical
computation (band power, RMS, kurtosis, etc.) — no learned parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec

# Allowed feature names (for input validation).
ALLOWED_FEATURES = {
    "rms_uv",
    "std_uv",
    "kurtosis",
    "peak_to_peak_uv",
    "line_length_uv_s",
    "hjorth_mobility",
    "zero_crossing_rate_hz",
    "delta_power",
    "theta_power",
    "alpha_power",
    "sigma_power",
    "beta_power",
    "gamma_power",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "sigma_relative",
    "beta_relative",
    "gamma_relative",
    "theta_alpha_ratio",
    "slow_fast_ratio",
}

# Hard output budget (tokens).  Context-budget management is a tested
# capability: an over-budget call is REJECTED with an actionable error
# (the agent must rescope) rather than silently truncated.  Uniform across
# all models so the tool's behaviour is a property of the benchmark, not of
# any model's context window.
MAX_OBS_TOKENS = 8000
_BYTES_PER_TOKEN = 4  # matches attach_response_meta's heuristic

# Extended bands including sigma (for sleep spindle detection).
_BANDS_EXT: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class ComputeWindowedFeaturesTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_windowed_features",
            version="0.1.0",
            description=(
                "Compute features in sliding windows across the recording, "
                "returning a per-window time-series for each channel. "
                "Available features: "
                "time-domain (rms_uv, std_uv, kurtosis, peak_to_peak_uv, "
                "line_length_uv_s, hjorth_mobility, zero_crossing_rate_hz), "
                "spectral absolute power (delta_power 1-4 Hz, theta_power "
                "4-8 Hz, alpha_power 8-13 Hz, sigma_power 11-16 Hz, "
                "beta_power 13-30 Hz, gamma_power 30-45 Hz), "
                "spectral relative (delta_relative ... gamma_relative), "
                "ratios (theta_alpha_ratio, slow_fast_ratio). "
                "Returns one row per window with columns: t_start, t_end, "
                "and the requested features.\n"
                "OUTPUT BUDGET: this tool enforces a hard 8000-token cap on "
                "its response. Output size = n_channels x n_windows rows "
                "(n_windows = (tmax-tmin)/step_sec; whole record if tmin/tmax "
                "omitted), at roughly 6-7 tokens per requested field per row. "
                "If a call would exceed 8000 tokens it is REJECTED with an "
                "error suggesting a smaller scope (it does NOT silently "
                "truncate, and nothing is computed). You must scope the call "
                "to fit: pick 1-3 channels, use tmin/tmax to analyse a time "
                "chunk, and/or use a coarser window_sec/step_sec. "
                "`_meta.approx_response_tokens` reports the actual size on "
                "success.\n"
                "\n"
                "## Applicability\n"
                "Works on any channel count incl. single-channel. Unknown "
                "channels, an empty/inverted window, or a window/step too small "
                "for the sampling rate return a clear error."
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
                        "description": (
                            "Channels to analyze. If omitted, uses ALL EEG "
                            "channels (multiplies output size). Prefer 1-3 "
                            "channels."
                        ),
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Feature names to compute per window. "
                            "If omitted, defaults to "
                            "[rms_uv, kurtosis, delta_relative, theta_relative, "
                            "alpha_relative, beta_relative, gamma_relative]."
                        ),
                    },
                    "window_sec": {
                        "type": "number",
                        "description": (
                            "Window length in seconds. Default 5; a coarser "
                            "window (e.g. 30) returns far fewer rows."
                        ),
                    },
                    "step_sec": {
                        "type": "number",
                        "description": (
                            "Step size in seconds (controls overlap). "
                            "Default equals window_sec (no overlap)."
                        ),
                    },
                    "tmin": {
                        "type": "number",
                        "description": (
                            "Start time in seconds. Omitting tmin/tmax "
                            "processes the whole record (large on long files)."
                        ),
                    },
                    "tmax": {
                        "type": "number",
                        "description": "End time in seconds.",
                    },
                },
                "required": ["record_id"],
            },
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_window_features(
        segment: np.ndarray,
        sfreq: float,
        requested: set[str],
    ) -> dict[str, float]:
        """Compute all requested features for a single-channel segment."""
        from eeg_agent_bench.tools.implementations._data_utils import (
            _round, integrate_power,
        )
        from scipy import signal as sig
        from scipy.stats import kurtosis as sp_kurtosis

        x = segment
        x_uv = x * 1e6
        n = len(x)
        duration = n / sfreq
        result: dict[str, float] = {}

        # ── Time-domain features ─────────────────────────────────────
        need_time = requested & {
            "rms_uv", "std_uv", "kurtosis", "peak_to_peak_uv",
            "line_length_uv_s", "hjorth_mobility", "zero_crossing_rate_hz",
        }
        if need_time:
            if "rms_uv" in requested:
                result["rms_uv"] = _round(float(np.sqrt(np.mean(x_uv ** 2))), 2)
            if "std_uv" in requested:
                result["std_uv"] = _round(float(np.std(x_uv)), 2)
            if "kurtosis" in requested:
                result["kurtosis"] = _round(float(sp_kurtosis(x_uv)), 3)
            if "peak_to_peak_uv" in requested:
                result["peak_to_peak_uv"] = _round(float(np.ptp(x_uv)), 1)
            if "line_length_uv_s" in requested:
                dx_uv = np.diff(x_uv)
                ll = float(np.sum(np.abs(dx_uv)) / max(duration, 1e-6))
                result["line_length_uv_s"] = _round(ll, 1)
            if "hjorth_mobility" in requested:
                dx = np.diff(x)
                var_x = float(np.var(x))
                var_dx = float(np.var(dx))
                mobility = float(np.sqrt(var_dx / max(var_x, 1e-30)))
                result["hjorth_mobility"] = _round(mobility, 4)
            if "zero_crossing_rate_hz" in requested:
                zc = float(
                    np.sum(np.diff(np.sign(x - np.mean(x))) != 0)
                    / max(duration, 1e-6)
                )
                result["zero_crossing_rate_hz"] = _round(zc, 2)

        # ── Spectral features ────────────────────────────────────────
        need_spectral = requested & {
            "delta_power", "theta_power", "alpha_power", "sigma_power",
            "beta_power", "gamma_power",
            "delta_relative", "theta_relative", "alpha_relative",
            "sigma_relative", "beta_relative", "gamma_relative",
            "theta_alpha_ratio", "slow_fast_ratio",
        }
        if need_spectral:
            nperseg = min(int(sfreq * 2), n)
            if nperseg < 4:
                # Window too short for spectral analysis
                for feat in need_spectral:
                    result[feat] = 0.0
            else:
                freqs, psd = sig.welch(x, fs=sfreq, nperseg=nperseg)

                band_abs: dict[str, float] = {}
                for band, (flo, fhi) in _BANDS_EXT.items():
                    mask = (freqs >= flo) & (freqs <= fhi)
                    if mask.any():
                        band_abs[band] = integrate_power(psd[mask], freqs[mask]) * 1e12
                    else:
                        band_abs[band] = 0.0

                total = sum(band_abs.values())
                total = max(total, 1e-20)

                for band in _BANDS_EXT:
                    key_abs = f"{band}_power"
                    if key_abs in requested:
                        result[key_abs] = _round(band_abs[band])

                for band in _BANDS_EXT:
                    key_rel = f"{band}_relative"
                    if key_rel in requested:
                        result[key_rel] = _round(band_abs[band] / total, 4)

                if "theta_alpha_ratio" in requested:
                    alpha_p = max(band_abs.get("alpha", 1e-20), 1e-20)
                    result["theta_alpha_ratio"] = _round(
                        band_abs.get("theta", 0) / alpha_p
                    )
                if "slow_fast_ratio" in requested:
                    slow = band_abs.get("delta", 0) + band_abs.get("theta", 0)
                    fast = max(
                        band_abs.get("alpha", 0) + band_abs.get("beta", 0),
                        1e-20,
                    )
                    result["slow_fast_ratio"] = _round(slow / fast)

        return result

    # ── main execute ─────────────────────────────────────────────────

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, get_eeg_channels, get_data_array, attach_response_meta,
        )

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        window_sec = arguments.get("window_sec", 5.0)
        step_sec = arguments.get("step_sec", window_sec)
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")

        features = arguments.get("features")
        if features is None:
            features = [
                "rms_uv", "kurtosis",
                "delta_relative", "theta_relative", "alpha_relative",
                "beta_relative", "gamma_relative",
            ]
        requested = set(features) & ALLOWED_FEATURES
        if not requested:
            return {
                "record_id": record_id,
                "error": (
                    f"No valid features requested. "
                    f"Allowed: {sorted(ALLOWED_FEATURES)}"
                ),
            }

        raw = load_raw(record_id, context)
        if channels is None:
            channels = get_eeg_channels(raw)
        valid_chs = [ch for ch in channels if ch in raw.ch_names]
        if not valid_chs:
            return {
                "record_id": record_id,
                "error": f"None of the requested channels exist: {channels}",
            }

        # Validate time window
        if tmin is not None and tmax is not None and tmax <= tmin:
            return {
                "record_id": record_id,
                "error": f"tmax ({tmax}) must be greater than tmin ({tmin}).",
            }

        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)
        n_channels, n_total = data.shape

        if n_total < 2:
            return {
                "record_id": record_id,
                "error": (
                    f"Too few samples ({n_total}) in the requested time window. "
                    f"Ensure tmin/tmax fall within the recording duration."
                ),
            }

        total_sec = n_total / sfreq

        actual_tmin = tmin if tmin is not None else 0.0

        win_samples = int(window_sec * sfreq)
        step_samples = int(step_sec * sfreq)
        if win_samples < 4 or step_samples < 1:
            return {
                "record_id": record_id,
                "error": "Window or step too small for the sampling rate.",
            }

        starts = list(range(0, n_total - win_samples + 1, step_samples))
        if not starts:
            starts = [0]
            win_samples = min(win_samples, n_total)

        # ── Pre-flight output-budget check (reject, do NOT truncate) ───
        # Estimate the response size from a representative row built with
        # the ACTUAL requested feature keys, then gate on the 8000-token
        # budget *before* computing anything.  Over-budget -> actionable
        # error so the agent rescopes (context-budget mgmt is in-scope).
        import json as _json_size
        _sample_row: dict[str, Any] = {f: -12.3456 for f in sorted(requested)}
        _sample_row["t_start"] = 0.0
        _sample_row["t_end"] = 0.0
        row_bytes = len(_json_size.dumps(_sample_row)) + 2  # +per-row comma/braces
        n_windows = len(starts)
        n_ch = len(valid_chs)
        est_bytes = n_windows * n_ch * row_bytes + 60 * n_ch  # +channel wrappers
        est_tokens = est_bytes // _BYTES_PER_TOKEN
        if est_tokens > MAX_OBS_TOKENS:
            max_rows = max(1, (MAX_OBS_TOKENS * _BYTES_PER_TOKEN) // row_bytes)
            max_windows = max(1, max_rows // n_ch)
            max_channels = max(1, max_rows // max(n_windows, 1))
            sugg_span_sec = round(max_windows * step_sec, 1)
            return {
                "record_id": record_id,
                "error": (
                    f"Output too large: this call would return {n_ch} channels "
                    f"x {n_windows} windows = {n_ch * n_windows} rows "
                    f"(~{int(est_tokens)} tokens), exceeding the "
                    f"{MAX_OBS_TOKENS}-token budget. Nothing was computed. "
                    f"Rescope to fit, e.g.: (a) use <= {max_channels} "
                    f"channel(s) at the current time span; or (b) keep "
                    f"<= {max_windows} windows by limiting the time range to a "
                    f"<= {sugg_span_sec}s chunk via tmin/tmax (record is "
                    f"{round(total_sec, 1)}s long); or (c) use a coarser "
                    f"window_sec / step_sec. Then call again on the chunk."
                ),
                "budget_tokens": MAX_OBS_TOKENS,
                "estimated_tokens": int(est_tokens),
                "n_channels": n_ch,
                "n_windows": n_windows,
            }

        channel_results: list[dict[str, Any]] = []
        for ch_idx, ch_name in enumerate(valid_chs):
            windows: list[dict[str, Any]] = []
            for s in starts:
                e = min(s + win_samples, n_total)
                segment = data[ch_idx, s:e]
                t_start = round(actual_tmin + s / sfreq, 2)
                t_end = round(actual_tmin + e / sfreq, 2)

                feats = self._compute_window_features(segment, sfreq, requested)
                feats["t_start"] = t_start
                feats["t_end"] = t_end
                windows.append(feats)

            channel_results.append({
                "channel": ch_name,
                "n_windows": len(windows),
                "windows": windows,
            })

        result: dict[str, Any] = {
            "record_id": record_id,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "time_range": [actual_tmin, round(actual_tmin + total_sec, 2)],
            "n_windows": len(starts),
            "features_computed": sorted(requested),
            "n_channels": len(valid_chs),
            "channels": channel_results,
        }
        # No silent truncation: over-budget calls are rejected by the
        # pre-flight check above, so a returned result is always complete.
        return attach_response_meta(
            result,
            n_rows=len(valid_chs) * len(starts),
            truncated=False,
            truncated_reason=None,
        )
