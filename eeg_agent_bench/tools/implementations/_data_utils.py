"""Shared EEG data loading, caching, and constants.

All tools use this module to resolve record_id → raw MNE object,
with a simple in-memory LRU cache to avoid re-reading .edf files
across multiple tool calls within the same scenario.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Standard frequency bands ─────────────────────────────────────────
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ── Standard 10-20 homologous channel pairs (left, right) ────────────
HOMOLOGOUS_PAIRS: list[tuple[str, str]] = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
    ("F7", "F8"),
    ("T3", "T4"),  # also called T7/T8
    ("T5", "T6"),  # also called P7/P8
]

# Alias map for alternate channel naming conventions
CHANNEL_ALIASES: dict[str, str] = {
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
    "T3": "T3", "T4": "T4",
    "T5": "T5", "T6": "T6",
}

# Hemisphere assignment for standard 10-20
HEMISPHERE_MAP: dict[str, str] = {
    "Fp1": "L", "Fp2": "R", "Fpz": "M",
    "F3": "L", "F4": "R", "Fz": "M", "F7": "L", "F8": "R",
    "C3": "L", "C4": "R", "Cz": "M",
    "P3": "L", "P4": "R", "Pz": "M",
    "O1": "L", "O2": "R", "Oz": "M",
    "T3": "L", "T4": "R", "T7": "L", "T8": "R",
    "T5": "L", "T6": "R", "P7": "L", "P8": "R",
    "A1": "L", "A2": "R",
}

REGION_MAP: dict[str, str] = {
    "Fp1": "frontal", "Fp2": "frontal", "Fpz": "frontal",
    "F3": "frontal", "F4": "frontal", "Fz": "frontal",
    "F7": "temporal", "F8": "temporal",
    "C3": "central", "C4": "central", "Cz": "central",
    "P3": "parietal", "P4": "parietal", "Pz": "parietal",
    "O1": "occipital", "O2": "occipital", "Oz": "occipital",
    "T3": "temporal", "T4": "temporal", "T7": "temporal", "T8": "temporal",
    "T5": "temporal", "T6": "temporal", "P7": "temporal", "P8": "temporal",
}


# ── Data loading ─────────────────────────────────────────────────────

def _resolve_data_path(record_id: str, context: dict[str, Any]) -> Path:
    """Resolve record_id to a filesystem path using runtime context."""
    record_paths = context.get("record_paths", {})
    if record_id in record_paths:
        return Path(record_paths[record_id])
    raise FileNotFoundError(
        f"No data path found for record_id='{record_id}'. "
        f"Available records: {list(record_paths.keys())}"
    )


def _load_bonn_txt(path: Path, record_id: str, context: dict[str, Any], preload: bool = True):
    """Load a Bonn single-channel text series as an MNE RawArray."""
    import mne

    metadata = context.get("record_metadata", {}).get(record_id, {})
    sfreq = float(metadata.get("sampling_rate", 173.61))
    ch_name = metadata.get("channel_names", ["EEG"])[0]
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 1:
        data = np.ravel(data)

    # Bonn text files are conventionally integer amplitudes in microvolts.
    data_volts = data[np.newaxis, :] * 1e-6
    info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types=["eeg"])
    return mne.io.RawArray(data_volts, info, verbose=False)


def _load_npy(path: Path, record_id: str, context: dict[str, Any], preload: bool = True):
    """Load a single/multi-channel ``.npy`` epoch as an MNE RawArray.

    Used by synthetic-epoch tasks (e.g. C2-Artifact / EEGdenoiseNet).  The
    array is stored in **microvolts**; sampling rate and channel names come
    from the scenario record metadata so no label is embedded in the signal.
    """
    import mne

    metadata = context.get("record_metadata", {}).get(record_id, {})
    sfreq = float(metadata.get("sampling_rate", 256.0))
    data = np.load(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_channels = data.shape[0]
    ch_names = metadata.get("channel_names")
    if not ch_names or len(ch_names) != n_channels:
        ch_names = [f"EEG{i + 1}" if n_channels > 1 else "EEG" for i in range(n_channels)]

    # Stored amplitudes are in microvolts; MNE works in volts internally.
    data_volts = np.asarray(data, dtype=float) * 1e-6
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data_volts, info, verbose=False)


# Simple module-level cache: {path_str: mne.io.Raw}
_raw_cache: dict[str, Any] = {}
_MAX_CACHE = 4


def load_raw(record_id: str, context: dict[str, Any], preload: bool = True):
    """Load an EEG recording as MNE Raw object, with caching.

    Returns:
        mne.io.Raw object.

    Raises:
        ImportError: if MNE is not installed.
        FileNotFoundError: if the data file does not exist.
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "MNE-Python is required for EEG tool execution. "
            "Install with: pip install mne"
        )

    path = _resolve_data_path(record_id, context)
    path_str = str(path)

    if path_str in _raw_cache:
        return _raw_cache[path_str]

    if not path.exists():
        # Do NOT echo the filesystem path: directory names can leak labels
        # (e.g. ".../bonn/S/..." => epileptic) and the error string is
        # surfaced to the model by the tool env.  Reference the record_id only.
        raise FileNotFoundError(f"EEG data file not found for record_id={record_id!r}")

    mne.set_log_level("ERROR")
    suffix = path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=preload, verbose=False)
    elif suffix in (".fif", ".fif.gz"):
        raw = mne.io.read_raw_fif(path, preload=preload, verbose=False)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)
    elif suffix == ".txt":
        raw = _load_bonn_txt(path, record_id, context, preload=preload)
    elif suffix == ".npy":
        raw = _load_npy(path, record_id, context, preload=preload)
    else:
        raw = mne.io.read_raw(path, preload=preload, verbose=False)

    # Evict oldest if cache full
    if len(_raw_cache) >= _MAX_CACHE:
        oldest_key = next(iter(_raw_cache))
        del _raw_cache[oldest_key]

    _raw_cache[path_str] = raw
    return raw


def get_eeg_channels(raw) -> list[str]:
    """Return only EEG channel names from a Raw object."""
    import mne
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    return [raw.ch_names[i] for i in eeg_picks]


def get_data_array(
    raw,
    channels: list[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, float]:
    """Extract data array from Raw.

    Returns:
        (data, sfreq) where data shape is (n_channels, n_samples).
    """
    picks = channels if channels else get_eeg_channels(raw)
    # Filter to channels that actually exist
    valid = [ch for ch in picks if ch in raw.ch_names]
    if not valid:
        raise ValueError(f"None of the requested channels exist: {picks}")

    sfreq = raw.info["sfreq"]
    start = int(tmin * sfreq) if tmin is not None else 0
    stop = int(tmax * sfreq) if tmax is not None else None

    data = raw.get_data(picks=valid, start=start, stop=stop)
    return data, sfreq


# ── Shared input validators (fail loud, never silent-garbage) ────────
# On inapplicable or degenerate input, tools return structured errors rather
# input a tool must return a clear ``error`` dict — never an uncaught
# exception (cryptic numpy message) and never a plausible-but-empty "ok".
# These helpers centralise the two most common boundary checks so every
# tool reports them identically.

def resolve_valid_channels(
    raw, channels: list[str] | None
) -> tuple[list[str], dict[str, Any] | None]:
    """Resolve requested channels to existing ones.

    Returns ``(valid_channels, None)`` on success, or ``([], error_dict)``
    when *none* of the requested channels exist (so callers fail loudly
    instead of returning an empty-but-"ok" result).
    """
    picks = channels if channels else get_eeg_channels(raw)
    valid = [ch for ch in picks if ch in raw.ch_names]
    if not valid:
        preview = ", ".join(raw.ch_names[:20]) + (" …" if len(raw.ch_names) > 20 else "")
        return [], {
            "error": (
                f"None of the requested channels exist in this recording: "
                f"{channels}. Available channels: {preview}"
            )
        }
    return valid, None


def validate_time_window(
    raw, tmin: float | None, tmax: float | None, *, min_samples: int = 2
) -> dict[str, Any] | None:
    """Validate a requested ``[tmin, tmax]`` window against the recording.

    Returns ``None`` if the window is usable, else an ``error`` dict.
    Catches: ``tmax <= tmin``; windows starting past the recording; and
    windows that resolve to fewer than ``min_samples`` samples (the root
    cause of the empty-array crashes in PSD / temporal / transient tools).
    """
    sfreq = float(raw.info["sfreq"])
    dur = raw.n_times / sfreq
    if tmin is not None and tmax is not None and tmax <= tmin:
        return {"error": f"tmax ({tmax}) must be greater than tmin ({tmin})."}
    start = int((tmin or 0.0) * sfreq)
    stop = raw.n_times if tmax is None else min(int(tmax * sfreq), raw.n_times)
    n = stop - start
    if start >= raw.n_times or n < min_samples:
        return {
            "error": (
                f"Requested time window [{tmin}, {tmax}] s yields too few samples "
                f"({max(n, 0)}) within the {dur:.2f}s recording. "
                f"Choose a window inside [0, {dur:.2f}] s."
            )
        }
    return None


def _round(val: float, decimals: int = 4) -> float:
    """Round a float for JSON output."""
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return round(float(val), decimals)


def integrate_power(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate spectral power with NumPy 1.x/2.x compatibility."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# ── Response size meta (Layer-2 of tool size contract) ──────────────
# Every tool wraps its successful return through this helper so that the LLM sees
# a uniform `_meta` block at the top of the JSON, exposing how many
# tokens it just paid for the observation it is about to read.
#
# Design rules:
#   1. `_meta` MUST be the first key, so it shows up at the head of
#      long observation strings where LLMs attend most.
#   2. The helper does NOT mutate, summarize, or truncate the body.
#      It is a pure observer; truncation decisions remain inside each
#      tool's own logic (e.g. compute_windowed_features.MAX_WINDOWS).
#   3. `approx_response_*` are estimates from JSON char count; they
#      will be slightly smaller than the actual provider tokenisation
#      but are stable enough for the model to plan budget.
#
# Used by all 10 official tools.

import json as _json

_BYTES_PER_TOKEN_HEURISTIC = 4   # ~4 chars/token average for English+JSON

def attach_response_meta(
    payload: dict[str, Any],
    *,
    n_rows: int | None = None,
    truncated: bool = False,
    truncated_reason: str | None = None,
) -> dict[str, Any]:
    """Prepend a `_meta` size-summary block to a tool response.

    Returns a NEW dict whose first key is `_meta`; the rest of `payload`
    is preserved verbatim (modulo any pre-existing `_meta` key, which
    is dropped — callers must not pre-populate it).

    Parameters
    ----------
    payload
        The tool's normal success-return dict.
    n_rows
        Optional row count if the tool has a natural row concept
        (e.g. n_windows * n_channels).  Reported as-is when provided.
    truncated
        ``True`` if the tool downsampled / capped the response.
    truncated_reason
        Free-text explanation of the cap, e.g. ``"n_windows>2000"``.

    Returns
    -------
    dict
        ``{"_meta": {...}, **payload}``.  The estimated bytes are
        computed from the **post-meta** body (so `approx_response_bytes`
        excludes the meta itself, keeping the number stable when the
        tool's row schema does not change).
    """
    body = {k: v for k, v in payload.items() if k != "_meta"}
    try:
        body_serialised = _json.dumps(body, ensure_ascii=False, default=str)
        approx_bytes = len(body_serialised)
    except (TypeError, ValueError):
        # Defensive: never let meta serialisation break the tool.
        approx_bytes = -1

    meta: dict[str, Any] = {
        "approx_response_bytes": approx_bytes,
        "approx_response_kb": round(approx_bytes / 1024, 1) if approx_bytes >= 0 else None,
        "approx_response_tokens": (
            approx_bytes // _BYTES_PER_TOKEN_HEURISTIC if approx_bytes >= 0 else None
        ),
        "truncated": bool(truncated),
    }
    if n_rows is not None:
        meta["n_rows"] = int(n_rows)
    if truncated_reason:
        meta["truncated_reason"] = str(truncated_reason)

    return {"_meta": meta, **body}
