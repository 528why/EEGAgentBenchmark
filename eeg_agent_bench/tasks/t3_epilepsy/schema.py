"""T3-Epilepsy output schemas.

The ACNS §4 Impression field is binary at the record level.  Different
datasets use slightly different label vocabularies; we keep them as
separate schemas so each scenario carries the exact enum the agent must
output.
"""

from __future__ import annotations

# Bonn (sets Z, O = healthy surface; S = ictal intracranial).  We use
# "normal" / "epileptic" because Bonn does not provide the ACNS-style
# "abnormal" label and the dataset's epileptic class is unambiguously
# ictal activity rather than a generic "abnormality".
T3_EPILEPSY_BONN_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["normal", "epileptic"],
            "description": "EEG classification: normal or epileptic activity.",
        },
        "findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supporting EEG findings for the classification.",
        },
    },
    "required": ["classification"],
}
