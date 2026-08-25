"""T6-Sleep output schema.

Sleep staging output schema:

- Final answer carries the hypnogram in **run-length encoding** (one
  segment per (start_epoch, end_epoch, stage) tuple) — token-friendly
  while preserving the full epoch-level information.  The evaluator
  expands the RLE back to a per-epoch label sequence before scoring.
- A separate ``architecture`` block reports REM%, N3%, and
  sleep_efficiency (= TST / TIB).

Stage enum is the AASM 5-class set (W / N1 / N2 / N3 / REM).
"""

from __future__ import annotations

T6_SLEEP_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypnogram_rle": {
            "type": "array",
            "description": (
                "Run-length-encoded hypnogram: each item describes one "
                "contiguous block of identical stage labels.  Epoch indices "
                "are 0-based, inclusive on both ends.  Segments must cover "
                "the full recording (n_epochs_total)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "start_epoch": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "First 30-s epoch in the segment (inclusive).",
                    },
                    "end_epoch": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Last 30-s epoch in the segment (inclusive).",
                    },
                    "stage": {
                        "type": "string",
                        "enum": ["W", "N1", "N2", "N3", "REM"],
                        "description": "AASM 5-class sleep stage.",
                    },
                },
                "required": ["start_epoch", "end_epoch", "stage"],
            },
        },
        "architecture": {
            "type": "object",
            "description": (
                "Per-night architecture summary as fractions in [0, 1] "
                "(NOT percentages)."
            ),
            "properties": {
                "REM_pct": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "REM epochs / total epochs.",
                },
                "N3_pct": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "N3 epochs / total epochs.",
                },
                "sleep_efficiency": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Total sleep time / total recording time = "
                        "(non-W epochs) / total epochs."
                    ),
                },
            },
            "required": ["REM_pct", "N3_pct", "sleep_efficiency"],
        },
    },
    "required": ["hypnogram_rle", "architecture"],
}


__all__ = ["T6_SLEEP_OUTPUT_SCHEMA"]
