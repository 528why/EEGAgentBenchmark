"""T4-Dementia: record-level cohort discrimination (non-diagnostic).

The agent identifies which research cohort a recording belongs to from EEG
features. This is explicitly not a clinical diagnosis task.

Dataset: ds004504 (AD / FTD / HC, eyes-closed resting-state EEG, 19 channels).
"""

from eeg_agent_bench.tasks.t4_dementia.prompt import T4DementiaTask

__all__ = ["T4DementiaTask"]
