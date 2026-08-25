"""Core type definitions for EEG-AgentBench."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Rollout State Machine ──────────────────────────────────────────────

class RolloutState(str, enum.Enum):
    """State machine for a single scenario rollout.

    PENDING -> RUNNING -> TOOL_CALLING -> RUNNING
                       -> MEMORY_UPDATING -> RUNNING
                       -> COMPLETED
                       -> FAILED
                       -> MAX_TURNS_EXCEEDED
    """

    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    MEMORY_UPDATING = "memory_updating"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"


class ActionType(str, enum.Enum):
    """Agent action types."""

    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"
    MEMORY_UPDATE = "memory_update"
    INVALID = "invalid_action"


class TrajectoryEventType(str, enum.Enum):
    """Trajectory event types for replay and analysis."""

    PROMPT = "prompt"
    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    MEMORY_UPDATE = "memory_update"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


# ── Access Mode ────────────────────────────────────────────────────────

class AccessMode(str, enum.Enum):
    METADATA_ONLY_WITHOUT_TOOLS = "metadata_only_without_tools"
    METADATA_WITH_TOOLS = "metadata_with_tools"
    MULTI_SUBJECT = "multi_subject"
    SEQUENTIAL_RECORDS = "sequential_records"


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class RecordRef:
    """Reference to a single EEG record within a scenario."""
    record_id: str
    role: str = "target"  # target | reference | query
    order: int | None = None  # For T6 sequential tasks
    data_path: str = ""  # Path to the raw EEG file (e.g. .edf)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata typically: {sampling_rate, channels, duration_sec, montage, ...}


@dataclass
class SubjectRef:
    """Reference to a subject (for T4 cross-subject tasks)."""
    subject_id: str
    role: str = "query"  # reference | query
    label_visible_to_agent: str | None = None


@dataclass
class AccessConfig:
    """Controls how agent can interact with data and tools."""
    mode: AccessMode = AccessMode.METADATA_ONLY_WITHOUT_TOOLS
    allowed_tools: list[str] = field(default_factory=list)
    forward_only: bool = False
    max_turns: int = 1000


@dataclass
class EvaluatorConfig:
    """Evaluator specification for a scenario."""
    primary: str = ""
    secondary: list[str] = field(default_factory=list)


@dataclass
class ReleaseConfig:
    """Release metadata for a scenario."""
    split: str = "candidate"  # candidate | train | test
    include_gold_in_release: bool = False


@dataclass
class Scenario:
    """A single evaluation scenario — the minimal verifiable unit.

    General enough for T1-T6: uses records/subjects lists + task-specific input.
    """

    scenario_id: str
    task_id: str
    task_name: str
    dataset: str
    dataset_version: str = ""
    records: list[RecordRef] = field(default_factory=list)
    subjects: list[SubjectRef] = field(default_factory=list)
    input: dict[str, Any] = field(default_factory=dict)
    access: AccessConfig = field(default_factory=AccessConfig)
    expected_output_schema: dict[str, Any] = field(default_factory=dict)
    gold_private: dict[str, Any] = field(default_factory=dict)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        records = []
        for r in d.get("records", []):
            # Filter to known RecordRef fields
            rec_kwargs: dict[str, Any] = {"record_id": r["record_id"]}
            if "role" in r:
                rec_kwargs["role"] = r["role"]
            if "order" in r:
                rec_kwargs["order"] = r["order"]
            if "data_path" in r:
                rec_kwargs["data_path"] = r["data_path"]
            if "metadata" in r:
                rec_kwargs["metadata"] = r["metadata"]
            records.append(RecordRef(**rec_kwargs))
        subjects = [SubjectRef(**s) for s in d.get("subjects", [])]
        access = AccessConfig(**d["access"]) if "access" in d else AccessConfig()
        if isinstance(access.mode, str):
            access.mode = AccessMode(access.mode)
        evaluator = EvaluatorConfig(**d["evaluator"]) if "evaluator" in d else EvaluatorConfig()
        release = ReleaseConfig(**d["release"]) if "release" in d else ReleaseConfig()
        return cls(
            scenario_id=d["scenario_id"],
            task_id=d["task_id"],
            task_name=d.get("task_name", ""),
            dataset=d["dataset"],
            dataset_version=d.get("dataset_version", ""),
            records=records,
            subjects=subjects,
            input=d.get("input", {}),
            access=access,
            expected_output_schema=d.get("expected_output_schema", {}),
            gold_private=d.get("gold_private", {}),
            evaluator=evaluator,
            release=release,
        )

    def _record_to_dict(self, r: RecordRef) -> dict[str, Any]:
        d: dict[str, Any] = {"record_id": r.record_id, "role": r.role}
        if r.order is not None:
            d["order"] = r.order
        if r.data_path:
            d["data_path"] = r.data_path
        if r.metadata:
            d["metadata"] = r.metadata
        return d

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "records": [self._record_to_dict(r) for r in self.records],
            "subjects": [{"subject_id": s.subject_id, "role": s.role, **({"label_visible_to_agent": s.label_visible_to_agent} if s.label_visible_to_agent else {})} for s in self.subjects],
            "input": self.input,
            "access": {
                "mode": self.access.mode.value,
                "allowed_tools": self.access.allowed_tools,
                "forward_only": self.access.forward_only,
                "max_turns": self.access.max_turns,
            },
            "expected_output_schema": self.expected_output_schema,
            "gold_private": self.gold_private,
            "evaluator": {
                "primary": self.evaluator.primary,
                "secondary": self.evaluator.secondary,
            },
            "release": {
                "split": self.release.split,
                "include_gold_in_release": self.release.include_gold_in_release,
            },
        }


# ── Agent Action & Response ───────────────────────────────────────────

@dataclass
class ToolCallRequest:
    """A single tool call request from the agent."""
    tool_call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Normalised agent output — one of three action types."""
    action_type: ActionType
    # For FINAL_ANSWER
    final_answer: dict[str, Any] | None = None
    # For TOOL_CALL
    tool_calls: list[ToolCallRequest] | None = None
    # For MEMORY_UPDATE
    memory_content: str | None = None
    # Raw assistant message (for trajectory logging)
    raw_message: dict[str, Any] | None = None
    thought: str | None = None
    # Separate reasoning trace returned by providers that expose chain-of-
    # thought in a dedicated field (e.g. OpenRouter ``message.reasoning``).
    # This is distinct from ``thought``, which is extracted from the model's
    # visible ``content`` text.
    reasoning: str | None = None

    @property
    def is_final_answer(self) -> bool:
        return self.action_type == ActionType.FINAL_ANSWER

    @property
    def has_tool_calls(self) -> bool:
        return self.action_type == ActionType.TOOL_CALL and bool(self.tool_calls)

    @property
    def is_memory_update(self) -> bool:
        return self.action_type == ActionType.MEMORY_UPDATE


@dataclass
class AgentResponse:
    """Full response from agent adapter, including metadata."""
    action: AgentAction
    raw_response: Any = None
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


# ── Observation ───────────────────────────────────────────────────────

@dataclass
class Observation:
    """Environment observation returned to the agent."""
    status: str  # ok | error | tool_not_available | invalid_action
    content: Any = None
    tool_call_id: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    def to_message(self) -> dict[str, Any]:
        """Convert observation to a chat message for history.

        Uses ``role: "user"`` with a ``[Tool Result]`` prefix so that the
        message works with **all** LLM providers — including those that
        reject ``role: "tool"`` unless the preceding assistant message
        used native API-level ``tool_calls`` (e.g. DeepSeek, Qwen via
        third-party proxies).

        """
        content_str = str(self.content) if self.content else self.error or ""
        return {
            "role": "user",
            "content": f"[Tool Result]\n{content_str}",
        }

# ── Eval Result ───────────────────────────────────────────────────────

@dataclass
class EvalScore:
    """Score for a single metric."""
    metric: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation result for a single scenario run."""
    scenario_id: str
    status: str  # completed | failed | max_turns_exceeded
    prediction: dict[str, Any] = field(default_factory=dict)
    scores: list[EvalScore] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # An *invalid* run aborted on an error (agent/API/tool-harness failure,
    # e.g. context-overflow) and produced no genuine answer.  It is NOT
    # scored and MUST be excluded from metric denominators (counted/reported
    # separately) so a transient empty response can be retried safely.
    invalid: bool = False

    def to_dict(self, *, redact_gold: bool = False) -> dict[str, Any]:
        """Serialise to dict.

        Args:
            redact_gold: If ``True``, strip gold-bearing keys from
                ``score_details`` so the output is safe for leaderboard /
                public export.  The ``score`` (numeric values only) is
                always included.
        """
        if redact_gold:
            from eeg_agent_bench.evaluators.base import redact_details
            details = {s.metric: redact_details(s.details) for s in self.scores}
        else:
            details = {s.metric: s.details for s in self.scores}

        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "invalid": self.invalid,
            "prediction": self.prediction,
            "score": {s.metric: s.value for s in self.scores},
            "score_details": details,
            "usage": self.usage,
            "error": self.error,
        }


# ── Trajectory Entry ─────────────────────────────────────────────────

@dataclass
class TrajectoryEntry:
    """A single event in the trajectory log."""
    scenario_id: str
    step: int
    event_type: TrajectoryEventType
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    agent_message: dict[str, Any] | None = None
    tool_call: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "step": self.step,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
        }
        if self.agent_message:
            d["agent_message"] = self.agent_message
        if self.tool_call:
            d["tool_call"] = self.tool_call
        if self.observation:
            d["observation"] = self.observation
        if self.error:
            d["error"] = self.error
        return d


# ── Tool Spec ─────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """Specification of a single tool in the registry."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, bool] = field(default_factory=lambda: {"can_access_gold": False, "can_access_raw_signal": True})

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
