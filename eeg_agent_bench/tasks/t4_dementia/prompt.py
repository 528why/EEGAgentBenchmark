"""T4-Dementia prompt — cohort discrimination, explicitly NON-diagnostic.

This prompt differs from T3-Epilepsy in three ways:

1. **Framing**: not a clinical Impression — agent is asked which research
   cohort the recording belongs to.  We make this explicit so the agent
   does not over-interpret the dataset.
2. **Analysis hints**: cohort separations (AD/FTD/HC, emotion classes)
   are typically driven by *statistical* features (band-power
   distributions, asymmetry, connectivity) rather than by any single
   waveform or transient.  The prompt steers the agent toward those
   tools.
3. **Pragmatic-EM reminder**: the ``## Important`` block in the user
   message explicitly tells the agent that the output is a cohort
   identifier, not a diagnostic label.  This mirrors the bench's
   out-of-scope declaration in 0521_v1.md §七 #9.
"""

from __future__ import annotations

from eeg_agent_bench.tasks.classification import ClassificationTask
from eeg_agent_bench.types import Scenario

# ── System prompt template ────────────────────────────────────────────

T4_DEMENTIA_SYSTEM_TEMPLATE = """\
You are an expert EEG analyst agent.  Your task is **research-level \
cohort discrimination**: given one EEG recording, decide which cohort it \
belongs to, choosing exactly one of the labels listed in the Expected \
Output Schema (provided in the user message).  You are NOT asked to \
provide a clinical diagnosis — your output is a cohort identifier, not a \
diagnostic claim.

## Rules
1. Cohort separations are typically driven by **statistical EEG \
features**: relative band power, spectral slope, hemispheric asymmetry, \
frontal/posterior gradients, channel correlation patterns.  Plan \
analysis accordingly — single transients are usually not the right \
evidence here.
2. Ground every conclusion in **signal-level evidence** from the tools.
3. Do NOT fabricate findings that the tools did not return.
4. If tools are available, plan a focused analysis: consider the \
clinical context first (age, recording protocol), then choose tools that \
probe spectral distribution, hemispheric asymmetry, and \
anterior/posterior gradients.  Decide for yourself which pattern \
characterises which cohort.
5. If no tools are available, reason from clinical context and metadata, \
and state clearly that no signal-level analysis was performed.

{tool_overview}

## Data Access
The EEG record metadata is provided in the user message.  Pass the \
``record_id`` to any tool; tools handle data loading internally.

## Action Format
When you want to call a tool, include this JSON block:
```json
{{"action": "tool_call", "tool_name": "<name>", "arguments": {{...}}}}
```
You will receive the tool result, then continue your analysis.

When you are ready to give your final answer, include this JSON block:
```json
{{"action": "final_answer", "classification": "<one of the labels in the expected output schema>", "findings": ["..."]}}
```

You may write reasoning in plain text around the JSON block.  Use \
multiple tools to gather sufficient evidence before concluding.
"""


# ── Per-task user-prompt reminder ────────────────────────────────────

_NON_DIAGNOSTIC_REMINDER = (
    "## Important\n"
    "This is a cohort-discrimination task, **not** a clinical diagnosis. "
    "Your output identifies which research cohort the recording belongs "
    "to in this dataset; it does not claim that EEG alone can diagnose "
    "the underlying clinical condition."
)


class T4DementiaTask(ClassificationTask):
    """T4: dementia cohort discrimination, explicitly non-diagnostic."""

    @property
    def task_id(self) -> str:
        return "T4"

    @property
    def task_name(self) -> str:
        return "cohort_discrimination"

    @property
    def system_prompt_template(self) -> str:
        return T4_DEMENTIA_SYSTEM_TEMPLATE

    def build_extra_user_sections(self, scenario: Scenario) -> list[str]:
        # Inject the non-diagnostic reminder right before the expected
        # output schema in every T4-Dementia scenario, so the framing
        # travels with the scenario regardless of upstream prompt edits.
        return [_NON_DIAGNOSTIC_REMINDER]


__all__ = ["T4DementiaTask", "T4_DEMENTIA_SYSTEM_TEMPLATE"]
