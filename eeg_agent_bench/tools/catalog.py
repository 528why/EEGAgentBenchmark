"""Single source of truth for the analysis-tool catalog.

Design decision (fairness alignment): **all analysis tools are offered to
every task**.  Tool availability must NOT encode the intended solution
path — pruning tools per task would leak "which approach the designer
expects" (the same construct-validity problem as a prompt cheat-sheet) and
would remove tool *selection* from what we evaluate.

Therefore every task's ``access.allowed_tools`` is this full list.  The
burden moves to the tool layer: each tool must (a) describe its own
task-agnostic contract (what it computes / what data it accepts / what it
returns incl. size & truncation / how it reports "not applicable"), and
(b) fail loudly and transparently on inapplicable input rather than
returning plausible-but-wrong numbers.

Keep this list in sync with ``configs/tools.yaml`` (validated by
``tests``).  Order is the canonical display order used in prompts.
"""

from __future__ import annotations

ALL_ANALYSIS_TOOLS: list[str] = [
    "get_recording_info",
    "get_channel_list",
    "compute_noise_metrics",
    "compute_psd",
    "compute_band_power",
    "detect_transients",
    "compute_temporal_features",
    "compute_asymmetry",
    "compute_channel_correlation",
    "compute_windowed_features",
]

__all__ = ["ALL_ANALYSIS_TOOLS"]
