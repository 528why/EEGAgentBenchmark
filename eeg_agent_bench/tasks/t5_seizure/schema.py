"""T5-Seizure output schema.

Seizure detection output schema for long-record search and open output. The
agent reads one long scalp-EEG record and reports **all**
electrographic seizure events as a list of ``{onset_sec, offset_sec}``
intervals.  An **empty list means "no seizure in this record"** — which
is a valid (and common) answer; ~79% of CHB-MIT files are seizure-free.

The output is intentionally open-ended (variable-length event list), not
a single classification — this is what makes the task agentic (the agent
must search the whole record and decide where, if anywhere, seizures are).
"""

from __future__ import annotations

T5_SEIZURE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "seizure_events": {
            "type": "array",
            "description": (
                "All electrographic seizures found in the record, as "
                "time intervals in seconds from recording start.  Return "
                "an EMPTY list if no seizure is present."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "onset_sec": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Seizure onset, seconds from start.",
                    },
                    "offset_sec": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Seizure offset, seconds from start.",
                    },
                },
                "required": ["onset_sec", "offset_sec"],
            },
        },
    },
    "required": ["seizure_events"],
}


__all__ = ["T5_SEIZURE_OUTPUT_SCHEMA"]
