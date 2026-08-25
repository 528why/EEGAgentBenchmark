"""Robust RLE-hypnogram → per-epoch label expander.

The C3-Sleep agent submits its hypnogram as a list of run-length
segments::

    [
        {"start_epoch": 0,   "end_epoch": 79,  "stage": "W"},
        {"start_epoch": 80,  "end_epoch": 91,  "stage": "N1"},
        {"start_epoch": 92,  "end_epoch": 250, "stage": "N2"},
        ...
    ]

Real LLM outputs are messy, so this parser is *deliberately* tolerant:

- Missing segments / gaps  → fill with ``W``.
- Overlaps / out-of-order  → resolved by accepting the LAST written
  label for any contested epoch (later segments override earlier).
- Negative or non-integer indices → coerced; bad rows are skipped and
  surfaced as ``rle_warnings`` rather than raising.
- Out-of-range stage labels → mapped to ``W`` and logged.
- Tail beyond ``n_epochs_total`` → truncated.
- Empty / missing RLE → all-``W`` series (so downstream metrics still
  produce a sensible score rather than NaN).

The output is the canonical per-epoch label list the evaluators consume.
``rle_warnings`` accumulates human-readable strings so the evaluator can
report them in score details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AASM_LABELS: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
DEFAULT_FILL_LABEL: str = "W"

# Soft alias map for typical LLM noise in stage labels.
_STAGE_ALIASES: dict[str, str] = {
    "W": "W", "WAKE": "W", "0": "W",
    "N1": "N1", "1": "N1", "S1": "N1", "STAGE 1": "N1",
    "N2": "N2", "2": "N2", "S2": "N2", "STAGE 2": "N2",
    "N3": "N3", "3": "N3", "4": "N3", "S3": "N3", "S4": "N3",
    "STAGE 3": "N3", "STAGE 4": "N3",
    "R": "REM", "REM": "REM", "STAGE R": "REM",
}


def _canonicalise_stage(raw: Any) -> str | None:
    """Best-effort stage normaliser; returns None for un-normalisable input."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if s in _STAGE_ALIASES:
        return _STAGE_ALIASES[s]
    if s in AASM_LABELS:
        return s
    return None


@dataclass
class HypnogramParseResult:
    epoch_labels: list[str]
    n_epochs_total: int
    coverage: float  # fraction of epochs covered by valid segments (pre-fill)
    n_valid_segments: int
    n_segments_input: int
    rle_warnings: list[str] = field(default_factory=list)


def expand_rle_to_epochs(
    rle: list[Any],
    n_epochs_total: int,
    fill_label: str = DEFAULT_FILL_LABEL,
) -> HypnogramParseResult:
    """Expand a (possibly malformed) RLE list to a per-epoch label list.

    Args:
        rle: Raw list of RLE segments — items may be dicts with
            ``start_epoch``, ``end_epoch``, ``stage`` keys, or any
            other shape (which is then logged as a warning).
        n_epochs_total: Required output length.
        fill_label: Label used to pad gaps and unrecoverable segments.
            Default ``W`` for missing or malformed epochs.

    Returns:
        :class:`HypnogramParseResult` with a length-``n_epochs_total``
        label list and bookkeeping fields.
    """
    n_epochs_total = max(int(n_epochs_total), 0)
    labels: list[str | None] = [None] * n_epochs_total
    warnings: list[str] = []
    n_valid = 0

    if not isinstance(rle, list):
        warnings.append(f"hypnogram_rle is not a list (got {type(rle).__name__})")
        rle_iter: list[Any] = []
    else:
        rle_iter = rle

    for idx, seg in enumerate(rle_iter):
        if not isinstance(seg, dict):
            warnings.append(f"segment[{idx}] is not a dict; skipped")
            continue
        try:
            start = int(seg.get("start_epoch", -1))
            end = int(seg.get("end_epoch", -1))
        except (TypeError, ValueError):
            warnings.append(
                f"segment[{idx}] has non-integer epoch bounds; skipped"
            )
            continue
        stage = _canonicalise_stage(seg.get("stage"))
        if stage is None:
            warnings.append(
                f"segment[{idx}] stage '{seg.get('stage')!r}' "
                "unrecognised; skipped"
            )
            continue

        if start < 0 or end < 0:
            warnings.append(
                f"segment[{idx}] has negative bound (start={start}, "
                f"end={end}); skipped"
            )
            continue
        if end < start:
            warnings.append(
                f"segment[{idx}] end ({end}) < start ({start}); skipped"
            )
            continue

        if n_epochs_total == 0:
            continue

        # Truncate tail past n_epochs_total without an error.
        eff_end = min(end, n_epochs_total - 1)
        eff_start = min(start, n_epochs_total - 1)
        if eff_start != start or eff_end != end:
            warnings.append(
                f"segment[{idx}] truncated to [{eff_start}, {eff_end}] "
                f"(was [{start}, {end}])"
            )

        if eff_end < eff_start:
            continue

        for i in range(eff_start, eff_end + 1):
            labels[i] = stage
        n_valid += 1

    # Coverage = fraction of epochs that received a non-None label.
    if n_epochs_total > 0:
        n_covered = sum(1 for x in labels if x is not None)
        coverage = n_covered / n_epochs_total
    else:
        coverage = 0.0

    if n_epochs_total > 0 and coverage < 1.0:
        warnings.append(
            f"coverage={coverage:.3f}; filling {n_epochs_total - n_covered} "
            f"missing epoch(s) with '{fill_label}'"
        )

    final = [lbl if lbl is not None else fill_label for lbl in labels]

    return HypnogramParseResult(
        epoch_labels=final,
        n_epochs_total=n_epochs_total,
        coverage=coverage,
        n_valid_segments=n_valid,
        n_segments_input=len(rle_iter),
        rle_warnings=warnings,
    )


__all__ = [
    "AASM_LABELS",
    "DEFAULT_FILL_LABEL",
    "HypnogramParseResult",
    "expand_rle_to_epochs",
]
