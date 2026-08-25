"""Whole-night sleep staging and architecture summary.

The agent outputs a 30-second-epoch hypnogram in AASM 5-class
({W, N1, N2, N3, REM}) using a compact run-length encoding, plus the
per-night architecture summary (REM%, N3%, sleep efficiency).

Sleep staging output is *not* a single record-level classification, so
this task subclasses :class:`BaseTask` directly (not ``ClassificationTask``).

Dataset: Sleep-EDFx (Sleep Cassette + Sleep Telemetry, 197 PSGs total).
"""

from eeg_agent_bench.tasks.t6_sleep.prompt import T6SleepTask

__all__ = ["T6SleepTask"]
