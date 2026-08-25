"""Tool: compute_channel_correlation — inter-channel correlation matrix.

Computes Pearson correlation between arbitrary channel pairs over a
specified time window.  Essential for artifact type discrimination (T3):
  - Eye artifacts: Fp1-Fp2 high correlation (synchronous blinks)
  - Electrode pop: single channel uncorrelated with neighbours
  - Muscle artifact: focal, low inter-channel correlation

Design: purely non-parametric.  Pearson correlation is a deterministic
statistical computation — no learned parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputeChannelCorrelationTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_channel_correlation",
            version="0.1.0",
            description=(
                "Compute Pearson correlation coefficients between channel pairs "
                "over a specified time window.  Returns the upper triangle of "
                "the correlation matrix for the requested channels, plus a "
                "per-channel mean-absolute-correlation summary.\n"
                "\n"
                "## Output size\n"
                "n_rows = n_channels x (n_channels - 1) / 2 (upper triangle).\n"
                "  - 19-channel montage -> 171 pairs, ~12 KB, ~3k tokens.\n"
                "  - 64-channel montage -> 2016 pairs, ~140 KB, ~35k tokens.\n"
                "Tip: pass a focused channel subset to keep output concise; "
                "the matrix grows quadratically with n_channels.\n"
                "\n"
                "## Applicability\n"
                "Requires at least 2 valid channels and ≥2 samples in the window; "
                "single-channel data, unknown channels, or an empty/inverted "
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
                        "description": (
                            "Channels to include in the correlation matrix. "
                            "If omitted, uses all EEG channels. "
                            "Tip: pass a focused subset (e.g. frontal channels) "
                            "to keep output concise."
                        ),
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
        )

        record_id = arguments["record_id"]
        channels = arguments.get("channels")
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")

        raw = load_raw(record_id, context)
        if channels is None:
            channels = get_eeg_channels(raw)
        valid_chs = [ch for ch in channels if ch in raw.ch_names]
        if len(valid_chs) < 2:
            return {
                "record_id": record_id,
                "error": (
                    "Need at least 2 valid channels for correlation. "
                    f"Found: {valid_chs}"
                ),
            }

        # Validate time window
        if tmin is not None and tmax is not None and tmax <= tmin:
            return {
                "record_id": record_id,
                "error": f"tmax ({tmax}) must be greater than tmin ({tmin}).",
            }

        data, sfreq = get_data_array(raw, channels=valid_chs, tmin=tmin, tmax=tmax)
        n_channels, n_samples = data.shape

        if n_samples < 2:
            return {
                "record_id": record_id,
                "error": (
                    f"Too few samples ({n_samples}) in the requested time window. "
                    f"Ensure tmin/tmax fall within the recording duration."
                ),
            }

        # Identify constant (zero-variance) channels — correlation is
        # undefined for these and np.corrcoef would emit RuntimeWarning.
        variances = np.var(data, axis=1)
        constant_mask = variances < 1e-30  # True for flat-line channels
        constant_channels = [
            valid_chs[i] for i in range(n_channels) if constant_mask[i]
        ]

        # Compute correlation matrix with warnings suppressed for safety;
        # constant channels are handled explicitly below.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_matrix = np.corrcoef(data)  # shape (n_channels, n_channels)

        # Force constant-channel rows/cols to 0 (correlation undefined)
        for i in range(n_channels):
            if constant_mask[i]:
                corr_matrix[i, :] = 0.0
                corr_matrix[:, i] = 0.0

        # Replace any remaining NaN/Inf (shouldn't happen, but defensive)
        corr_matrix = np.where(
            np.isnan(corr_matrix) | np.isinf(corr_matrix), 0.0, corr_matrix
        )

        # Extract upper triangle as pair list
        pairs: list[dict[str, Any]] = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                pairs.append({
                    "channel_a": valid_chs[i],
                    "channel_b": valid_chs[j],
                    "correlation": _round(float(corr_matrix[i, j]), 4),
                })

        # Per-channel summary: mean absolute correlation with all others
        ch_summaries: list[dict[str, Any]] = []
        for i, ch in enumerate(valid_chs):
            row = corr_matrix[i, :]
            others = np.concatenate([row[:i], row[i+1:]])
            mean_abs_corr = float(np.mean(np.abs(others)))
            ch_summaries.append({
                "channel": ch,
                "mean_abs_correlation": _round(mean_abs_corr, 4),
            })

        result: dict[str, Any] = {
            "record_id": record_id,
            "time_window": [tmin, tmax],
            "n_channels": len(valid_chs),
            "channel_names": valid_chs,
            "n_pairs": len(pairs),
            "pairs": pairs,
            "channel_summary": ch_summaries,
        }
        if constant_channels:
            result["constant_channels"] = constant_channels
        return attach_response_meta(result, n_rows=len(pairs))
