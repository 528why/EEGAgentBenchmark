"""T4-Dementia output schemas.

These schemas describe **research cohort labels**, not clinical
diagnoses.  v3.1 explicitly frames this as an "out-of-scope-for-clinical"
task (Pragmatic EM boundary in 0521_v1.md §七 #9).
"""

from __future__ import annotations

# ds004504: 3-class cohort.  Resting-state, eyes-closed EEG.
T4_DEMENTIA_DS004504_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["AD", "FTD", "HC"],
            "description": (
                "Research cohort label for this recording: Alzheimer's "
                "disease (AD), frontotemporal dementia (FTD), or healthy "
                "control (HC).  This is a cohort-discrimination output, "
                "NOT a clinical diagnosis."
            ),
        },
        "findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supporting EEG features for the cohort label.",
        },
    },
    "required": ["classification"],
}
