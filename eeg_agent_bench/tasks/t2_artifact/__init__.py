"""Semi-synthetic artifact contamination recognition.

Built on EEGdenoiseNet data (clean EEG + scaled EOG/EMG artifact), the
agent uses measurement-only tools to classify a 2-second epoch as
``clean`` / ``ocular_contaminated`` / ``muscle_contaminated``.
"""

from eeg_agent_bench.tasks.t2_artifact.prompt import T2ArtifactTask

__all__ = ["T2ArtifactTask"]
