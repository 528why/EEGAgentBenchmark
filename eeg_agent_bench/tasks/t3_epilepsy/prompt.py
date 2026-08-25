"""T3-Epilepsy prompt — record-level Impression (normal vs abnormal).

This prompt is tuned for **single-record routine EEG impression**:
agent inspects one EEG and judges whether it is normal or abnormal,
listing the EEG-level evidence (e.g. background slowing, asymmetry,
epileptiform discharges, focal abnormalities).

The prompt is intentionally *different* from T4-Dementia: cohort
discrimination targets statistical group features (e.g. AD-related
slowing of posterior alpha) and is explicitly framed as non-diagnostic;
routine impression targets per-record judgement that maps directly to
ACNS Guideline 7 §4 Impression.
"""

from __future__ import annotations

from eeg_agent_bench.tasks.classification import ClassificationTask

# ── System prompt template ────────────────────────────────────────────
# Single ``{tool_overview}`` placeholder is filled at build time with the
# full tool reference.  All other braces are escaped as ``{{`` / ``}}``.

T3_EPILEPSY_SYSTEM_TEMPLATE = """\
You are an expert EEG analyst agent.  Your task is to read a single \
EEG recording and produce a record-level clinical **Impression**: \
classify the recording using exactly one of the labels listed in the \
Expected Output Schema (provided in the user message) and list the \
EEG-level evidence that supports your judgement.

## Rules
1. Your classification MUST be based on **signal-level evidence** \
obtained from the analysis tools (e.g. spectral features, transient \
morphology, temporal statistics, asymmetry).
2. Do NOT fabricate EEG findings that are not supported by tool results.
3. If tools are available, plan a focused analysis: consider the \
clinical context first, then pick tools that test specific hypotheses \
(e.g. "is the background normal?", "is there focal slowing?", \
"is there epileptiform activity?").
4. If no tools are available, reason from clinical context and metadata, \
and state clearly that no signal-level analysis was performed.
5. You may plan, call tools, update memory across multiple turns before \
submitting your final answer.
6. ``detect_transients`` returns **candidate** amplitude peaks, not \
confirmed epileptiform events.  Many candidates are normal physiological \
transients or artifacts.  Use background RMS, amplitude-to-RMS ratio, \
width and sharpness to judge clinical significance.

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


class T3EpilepsyTask(ClassificationTask):
    """T3: record-level normal-vs-epileptic screening."""

    @property
    def task_id(self) -> str:
        return "T3"

    @property
    def task_name(self) -> str:
        return "routine_impression"

    @property
    def system_prompt_template(self) -> str:
        return T3_EPILEPSY_SYSTEM_TEMPLATE


__all__ = ["T3EpilepsyTask", "T3_EPILEPSY_SYSTEM_TEMPLATE"]
