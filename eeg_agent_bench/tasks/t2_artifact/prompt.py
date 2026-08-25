"""Synthetic artifact contamination recognition prompt.

The task uses EEGdenoiseNet semi-synthetic data:
a short single-channel EEG epoch is either clean, or linearly mixed with an
ocular (EOG) or muscle (EMG) artifact source.  The agent uses
measurement-only tools to decide which.

The reference label is the synthetic contamination source.
"""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.tasks.classification import ClassificationTask

T2_ARTIFACT_SYSTEM_TEMPLATE = """\
You are an EEG signal-quality analyst agent.  You are given a short \
single-channel EEG epoch (about 2 seconds).  Decide whether the epoch is:

- ``clean`` — no artifact dominates the epoch; it looks like usable EEG \
for downstream reading.
- ``ocular_contaminated`` — dominated by an ocular / eye-movement (EOG) \
artifact (large, slow eye-movement deflections).
- ``muscle_contaminated`` — dominated by a muscle (EMG) artifact (fast, \
spiky muscle activity).

## Rules
1. Base your decision on **signal-level evidence** from the measurement \
tools (spectral distribution, band powers, temporal statistics, windowed \
features).  The tools report measurements only — they do NOT provide \
labels or contamination strength.  You must work out for yourself which \
spectral / temporal pattern indicates which source.
2. Do NOT fabricate features that are not supported by tool results.
3. You are NOT asked to report SNR or contamination strength — only the \
category.
4. **Calibration**: contamination is not always strong.  Assign a \
contamination class only when an artifact *clearly dominates* the epoch; \
if no artifact dominates, the epoch is ``clean``.
5. You may plan, call tools, and update memory across multiple turns \
before submitting your final answer.

{tool_overview}

## Data Access
The epoch metadata is provided in the user message.  Pass the \
``record_id`` to any tool; tools load the signal internally.

## Action Format
When you want to call a tool, include this JSON block:
```json
{{"action": "tool_call", "tool_name": "<name>", "arguments": {{...}}}}
```
You will receive the tool result, then continue your analysis.

When you are ready to answer, include this JSON block:
```json
{{"action": "final_answer", "classification": "<clean|ocular_contaminated|muscle_contaminated>", "findings": ["..."]}}
```

You may write reasoning in plain text around the JSON block.
"""


class T2ArtifactTask(ClassificationTask):
    """T2: synthetic artifact contamination recognition (3-class)."""

    @property
    def task_id(self) -> str:
        return "T2"

    @property
    def task_name(self) -> str:
        return "artifact_contamination"

    @property
    def system_prompt_template(self) -> str:
        return T2_ARTIFACT_SYSTEM_TEMPLATE

    def parse_final_answer(self, raw: Any) -> dict[str, Any]:
        """Accept both ``classification`` and ``label`` keys.

        The bench convention is ``classification``; ``label`` is accepted
        defensively because some models echo the user-message label space
        wording.  Output is always normalised to ``classification``.
        """
        out = super().parse_final_answer(raw)
        if not out.get("classification") and isinstance(raw, dict) and raw.get("label"):
            out["classification"] = raw.get("label", "")
        return out


__all__ = ["T2ArtifactTask", "T2_ARTIFACT_SYSTEM_TEMPLATE"]
