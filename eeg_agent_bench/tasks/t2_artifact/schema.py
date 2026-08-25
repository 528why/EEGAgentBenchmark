"""T2-Artifact output schema.

T2 (Synthetic Artifact Contamination Recognition) is a 3-class
single-label decision over a short EEG epoch:

    clean | ocular_contaminated | muscle_contaminated

The model does NOT report SNR or contamination strength — only the
category.  ``findings`` is optional, informational evidence (e.g.
"strong low-frequency drift", "broadband high-frequency power").
"""

from __future__ import annotations

T2_ARTIFACT_LABELS: list[str] = [
    "clean",
    "ocular_contaminated",
    "muscle_contaminated",
]

T2_ARTIFACT_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": list(T2_ARTIFACT_LABELS),
            "description": (
                "Signal-quality decision for the epoch: clean, "
                "ocular_contaminated (EOG / eye-movement), or "
                "muscle_contaminated (EMG)."
            ),
        },
        "findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supporting signal-level evidence (optional).",
        },
    },
    "required": ["classification"],
}

__all__ = ["T2_ARTIFACT_LABELS", "T2_ARTIFACT_OUTPUT_SCHEMA"]
