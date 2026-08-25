"""T1 output schema — single-answer 4-option MCQ.

The agent must output exactly one option letter (A/B/C/D).  The enum is
fixed, so a single shared schema is used for every T1 scenario.
"""

from __future__ import annotations

T1_MCQ_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
            "description": "The single correct option letter.",
        },
    },
    "required": ["answer"],
}

__all__ = ["T1_MCQ_OUTPUT_SCHEMA"]
