"""Prompt builder — assembles messages from scenario, tools, and response schema.

Design (text-based tool calling):
    **All tool information is placed in the system prompt** — name, description,
    full parameter spec, and a calling example.  This ensures compatibility with
    every LLM (OpenAI, Claude, Gemini, open-source via vLLM/SGLang, etc.) without
    relying on any provider-specific function-calling API.

    The model communicates actions via structured JSON in its text output:
    - Tool call  → ``{"action": "tool_call", "tool_name": "...", "arguments": {...}}``
    - Final answer → ``{"action": "final_answer", ...task-specific fields...}``

    The runner parses the JSON from the model's text response to drive the loop.
"""

from __future__ import annotations

import json
from typing import Any

from eeg_agent_bench.types import Scenario

# ── Tool categories (for prompt organisation) ─────────────────────────
TOOL_CATEGORIES: dict[str, list[str]] = {
    "Data Overview": [
        "get_recording_info",
        "get_channel_list",
    ],
    "Noise Metrics": [
        "compute_noise_metrics",
    ],
    "Spectral Analysis": [
        "compute_psd",
        "compute_band_power",
    ],
    "Transient Detection": [
        "detect_transients",
    ],
    "Temporal & Spatial Features": [
        "compute_temporal_features",
        "compute_asymmetry",
        "compute_channel_correlation",
    ],
    "Windowed / Time-Series Analysis": [
        "compute_windowed_features",
    ],
}


def _format_parameters(params: dict[str, Any]) -> str:
    """Format a JSON Schema ``parameters`` object into readable text."""
    props = params.get("properties", {})
    required = set(params.get("required", []))
    if not props:
        return "  (no parameters)"

    lines: list[str] = []
    for name, schema in props.items():
        ptype = schema.get("type", "any")
        # Array items
        if ptype == "array" and "items" in schema:
            ptype = f"array[{schema['items'].get('type', 'any')}]"
        # Enum
        if "enum" in schema:
            ptype = f"{ptype}, one of {schema['enum']}"
        # Default
        default = schema.get("default")
        default_str = f", default={json.dumps(default)}" if default is not None else ""
        # Required
        req_str = "required" if name in required else "optional"
        # Description
        desc = schema.get("description", "")
        desc_str = f" — {desc}" if desc else ""

        lines.append(f"    - `{name}` ({ptype}, {req_str}{default_str}){desc_str}")
    return "\n".join(lines)


# ── Uniform agent operating protocol (A-class guidance only) ─────────
# Appended verbatim to every task's tool overview, so all tasks receive
# byte-identical "how to operate the tools without breaking things"
# guidance. It contains only parameter, output-size, turn, and formatting
# guidance; it does not recommend task-specific tools or solution strategies.
AGENT_OPERATING_PROTOCOL = """\
## Using the Tools (operating protocol)
- Tools load data via `record_id`. You decide which tools to call, with which \
arguments, and in what order — selecting the right tool is part of the task.
- Observations can be large. Every tool result begins with a `_meta` block \
reporting `approx_response_tokens`; request focused channels and short \
`tmin`/`tmax` windows to keep each observation small, and process long \
recordings in chunks across turns so the running history stays within context.
- If a tool returns an `error` or reports it is "not applicable", read the \
message and adjust your arguments or choose a different tool — do not repeat \
the same failing call.
- You have a limited number of turns. Gather enough evidence, then submit your \
final answer. Emit exactly one `tool_call` JSON block per step."""


def build_tool_overview(tool_specs: list[dict[str, Any]]) -> str:
    """Build a **complete** tool reference from OpenAI-format tool specs.

    Includes for each tool: name, description, full parameter listing, and a
    calling example.  This is the single source of tool information for the
    model — no API-level function calling is assumed.
    """
    if not tool_specs:
        return (
            "## Available Tools\n"
            "No analysis tools are currently available.\n"
            "Reason based on the clinical context and record metadata provided."
        )

    # Parse specs
    tool_info: dict[str, dict[str, Any]] = {}
    for ts in tool_specs:
        fn = ts.get("function", {})
        name = fn.get("name", "")
        if name:
            tool_info[name] = {
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }

    available_names = set(tool_info.keys())
    lines: list[str] = []
    lines.append("## Available Tools")
    lines.append(f"You have access to **{len(available_names)}** EEG analysis tools.")
    lines.append("Tools handle data loading internally — just pass the `record_id`.\n")

    # ── Categorised listing with full parameter specs ─────────────
    for category, cat_tools in TOOL_CATEGORIES.items():
        present = [n for n in cat_tools if n in available_names]
        if not present:
            continue
        lines.append(f"### {category}\n")
        for n in present:
            info = tool_info[n]
            lines.append(f"**`{n}`** — {info['description']}")
            lines.append("  Parameters:")
            lines.append(_format_parameters(info["parameters"]))
            lines.append("")

    # Uncategorised tools
    uncategorised = available_names - {n for names in TOOL_CATEGORIES.values() for n in names}
    if uncategorised:
        lines.append("### Other\n")
        for n in sorted(uncategorised):
            info = tool_info[n]
            lines.append(f"**`{n}`** — {info['description']}")
            lines.append("  Parameters:")
            lines.append(_format_parameters(info["parameters"]))
            lines.append("")

    # ── Calling example ───────────────────────────────────────────
    example_name = next(iter(tool_info))
    example_params = tool_info[example_name]["parameters"]
    example_args: dict[str, Any] = {}
    for pname, pschema in example_params.get("properties", {}).items():
        if pname in example_params.get("required", []):
            example_args[pname] = f"<{pname}>"

    lines.append("### How to Call a Tool")
    lines.append("Respond with the following JSON to call a tool:")
    lines.append("```json")
    lines.append(json.dumps({
        "action": "tool_call",
        "tool_name": example_name,
        "arguments": example_args,
    }, indent=2))
    lines.append("```")
    lines.append("The tool result will be returned to you, and you can then "
                 "call more tools or submit your final answer.")

    # Uniform operating protocol (identical for every task).
    lines.append("")
    lines.append(AGENT_OPERATING_PROTOCOL)

    return "\n".join(lines)


class PromptBuilder:
    """Generic prompt assembly.

    Delegates task-specific content to the task's prompt module, but handles
    the generic structure: system prompt, tool declarations, response schema.
    """

    @staticmethod
    def build_initial_messages(
        scenario: Scenario,
        system_prompt: str,
        user_prompt: str,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble the initial message list for a scenario."""
        messages: list[dict[str, Any]] = []

        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        return messages
