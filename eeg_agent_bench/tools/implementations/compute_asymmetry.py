"""Tool: compute_asymmetry — left-right hemispheric differences."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class ComputeAsymmetryTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="compute_asymmetry",
            version="0.1.0",
            description=(
                "Compute left-right hemispheric asymmetry index (L−R)/(L+R) for "
                "standard 10-20 homologous channel pairs (Fp1-Fp2, F3-F4, C3-C4, "
                "P3-P4, O1-O2, F7-F8, T3-T4, T5-T6). Reports asymmetry in band "
                "power, total power, and amplitude. Positive values indicate "
                "left > right. "
                "Requires standard 10-20 homologous L/R pairs: if none are present "
                "(e.g. single-channel data or non-10-20 derivations) it returns "
                "``applicable: false`` with a not-applicable message instead of "
                "fabricating values. An empty/too-short window returns a clear "
                "error. Output is bounded (one row per available pair)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "EEG record identifier.",
                    },
                    "feature": {
                        "type": "string",
                        "enum": ["band_power", "amplitude", "all"],
                        "description": "Which feature to compute asymmetry for.",
                        "default": "all",
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
            load_raw, get_data_array, _round, BANDS, HOMOLOGOUS_PAIRS,
            CHANNEL_ALIASES, integrate_power, attach_response_meta,
            validate_time_window,
        )
        from scipy import signal as sig

        record_id = arguments["record_id"]
        feature = arguments.get("feature", "all")
        tmin = arguments.get("tmin")
        tmax = arguments.get("tmax")

        raw = load_raw(record_id, context)
        win_err = validate_time_window(
            raw, tmin, tmax, min_samples=max(8, int(raw.info["sfreq"] // 2))
        )
        if win_err:
            return {"record_id": record_id, **win_err}
        ch_names_set = set(raw.ch_names)

        # Also check for "EEG Fp1" style names
        def find_ch(target: str) -> str | None:
            if target in ch_names_set:
                return target
            canon = CHANNEL_ALIASES.get(target, target)
            if canon in ch_names_set:
                return canon
            prefixed = f"EEG {target}"
            if prefixed in ch_names_set:
                return prefixed
            return None

        pair_results = []
        for left_name, right_name in HOMOLOGOUS_PAIRS:
            left_ch = find_ch(left_name)
            right_ch = find_ch(right_name)
            if left_ch is None or right_ch is None:
                continue

            data, sfreq = get_data_array(raw, channels=[left_ch, right_ch], tmin=tmin, tmax=tmax)
            left_data = data[0]
            right_data = data[1]
            n_samples = len(left_data)
            nperseg = min(int(sfreq * 2), n_samples)

            pair_info: dict[str, Any] = {
                "left": left_name,
                "right": right_name,
            }

            # Amplitude asymmetry
            if feature in ("amplitude", "all"):
                left_rms = float(np.sqrt(np.mean(left_data ** 2))) * 1e6
                right_rms = float(np.sqrt(np.mean(right_data ** 2))) * 1e6
                denom = max(left_rms + right_rms, 1e-10)
                pair_info["amplitude_asymmetry"] = _round((left_rms - right_rms) / denom, 4)
                pair_info["left_rms_uv"] = _round(left_rms, 2)
                pair_info["right_rms_uv"] = _round(right_rms, 2)

            # Band power asymmetry
            if feature in ("band_power", "all"):
                freqs_l, psd_l = sig.welch(left_data, fs=sfreq, nperseg=nperseg)
                _, psd_r = sig.welch(right_data, fs=sfreq, nperseg=nperseg)

                band_asym = {}
                for band, (flo, fhi) in BANDS.items():
                    mask = (freqs_l >= flo) & (freqs_l <= fhi)
                    if not mask.any():
                        continue
                    lp = integrate_power(psd_l[mask], freqs_l[mask])
                    rp = integrate_power(psd_r[mask], freqs_l[mask])
                    denom = max(lp + rp, 1e-30)
                    band_asym[band] = _round((lp - rp) / denom, 4)

                pair_info["band_power_asymmetry"] = band_asym

                # Total power asymmetry
                total_l = integrate_power(psd_l, freqs_l)
                total_r = integrate_power(psd_r, freqs_l)
                pair_info["total_power_asymmetry"] = _round(
                    (total_l - total_r) / max(total_l + total_r, 1e-30), 4
                )

            pair_results.append(pair_info)

        # Explicit N/A: this tool requires standard 10-20 *monopolar*
        # homologous pairs (Fp1/Fp2, ...).  Datasets with single channels
        # (Bonn), pre-bipolar montages (CHB-MIT ``FP1-F7``), or non-10-20
        # derivations (Sleep-EDF ``EEG Fpz-Cz``) have no homologous pairs —
        # return a clear not-applicable message instead of an empty result
        # the agent might mistake for "no asymmetry".
        if not pair_results:
            return attach_response_meta(
                {
                    "record_id": record_id,
                    "applicable": False,
                    "n_pairs_analyzed": 0,
                    "pairs": [],
                    "reason": (
                        "No standard 10-20 homologous L/R pairs found in this "
                        "recording, so hemispheric asymmetry is not applicable. "
                        "This tool needs monopolar 10-20 channels (Fp1, F3, "
                        "C3, ...). The montage here is single-channel or "
                        "already bipolar/non-10-20."
                    ),
                    "available_channels": list(raw.ch_names),
                },
                n_rows=0,
            )

        return attach_response_meta(
            {
                "record_id": record_id,
                "applicable": True,
                "time_window": [tmin, tmax],
                "n_pairs_analyzed": len(pair_results),
                "pairs": pair_results,
            },
            n_rows=len(pair_results),
        )
