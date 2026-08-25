"""Regression tests for text-protocol and final-answer recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eeg_agent_bench.agents.adapters.openai_compatible import OpenAICompatibleAdapter
from eeg_agent_bench.agents.base import AgentAdapter
from eeg_agent_bench.config import AgentConfig, RolloutConfig
from eeg_agent_bench.envs.null_tool_env import NullToolEnvironment
from eeg_agent_bench.runners.run_scenario import run_scenario
from eeg_agent_bench.tools.registry import ToolRegistry
from eeg_agent_bench.types import (
    ActionType,
    AgentAction,
    AgentResponse,
    Scenario,
    ToolCallRequest,
)


class _Message:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


def _adapter():
    config = AgentConfig(name="test", provider="openai_compatible", model="test")
    return OpenAICompatibleAdapter(config)


def test_reasoning_only_text_is_invalid_not_final():
    action = _adapter()._parse_action(_Message("I am still merging the remaining epochs."))
    assert action.action_type == ActionType.INVALID
    assert action.final_answer is None


def test_empty_content_is_invalid_not_final():
    action = _adapter()._parse_action(_Message(""))
    assert action.action_type == ActionType.INVALID


def test_implicit_json_is_invalid_not_final():
    action = _adapter()._parse_action(_Message('{"record_id":"EEG_1","tmin":0,"tmax":60}'))
    assert action.action_type == ActionType.INVALID


def test_nested_tool_name_is_recovered():
    action = _adapter()._parse_action(_Message(
        '{"action":"tool_call","arguments":'
        '{"tool_name":"compute_windowed_features","record_id":"EEG_1","tmin":0,"tmax":60}}'
    ))
    assert action.action_type == ActionType.TOOL_CALL
    assert action.tool_calls[0].name == "compute_windowed_features"
    assert action.tool_calls[0].arguments["record_id"] == "EEG_1"
    assert "tool_name" not in action.tool_calls[0].arguments


def test_tool_marker_with_bare_arguments_is_recovered():
    action = _adapter()._parse_action(_Message(
        '<tool_call>{"tool_name":"compute_windowed_features",'
        '"record_id":"EEG_1","tmin":0,"tmax":60}</tool_call>'
    ))
    assert action.action_type == ActionType.TOOL_CALL
    assert action.tool_calls[0].name == "compute_windowed_features"
    assert action.tool_calls[0].arguments["record_id"] == "EEG_1"


def test_tagged_function_tool_call_is_recovered():
    action = _adapter()._parse_action(_Message(
        "<tool_call><function=get_recording_info>"
        "<parameter=record_id>EEG_1</parameter></function></tool_call>"
    ))
    assert action.action_type == ActionType.TOOL_CALL
    assert action.tool_calls[0].name == "get_recording_info"
    assert action.tool_calls[0].arguments == {"record_id": "EEG_1"}


def test_segmented_message_content_is_normalized():
    message = _Message("")
    message.content = [
        {"type": "text", "text": "Reasoning."},
        {"type": "text", "text": '{"action":"final_answer","answer":"A"}'},
    ]
    action = _adapter()._parse_action(message)
    assert action.action_type == ActionType.FINAL_ANSWER
    assert action.final_answer["answer"] == "A"


class _SequenceAgent(AgentAdapter):
    def __init__(self, actions):
        super().__init__(AgentConfig(name="sequence", provider="test", model="test"))
        self.actions = list(actions)
        self.seen_messages = []

    def generate(self, messages, tools=None, response_schema=None):
        self.seen_messages.append(list(messages))
        action = self.actions.pop(0)
        return AgentResponse(action=action, token_usage={})


class _NoopEvaluator:
    metric_name = "noop"

    def evaluate(self, scenario, prediction):
        return None


def _sleep_scenario():
    return Scenario(
        scenario_id="sleep_test",
        task_id="T6",
        task_name="sleep_staging",
        dataset="test",
        expected_output_schema={
            "type": "object",
            "properties": {
                "hypnogram_rle": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_epoch": {"type": "integer", "minimum": 0},
                            "end_epoch": {"type": "integer", "minimum": 0},
                            "stage": {"type": "string", "enum": ["W", "N1", "N2", "N3", "REM"]},
                        },
                        "required": ["start_epoch", "end_epoch", "stage"],
                    },
                },
                "architecture": {
                    "type": "object",
                    "properties": {
                        "REM_pct": {"type": "number", "minimum": 0, "maximum": 1},
                        "N3_pct": {"type": "number", "minimum": 0, "maximum": 1},
                        "sleep_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["REM_pct", "N3_pct", "sleep_efficiency"],
                },
            },
            "required": ["hypnogram_rle", "architecture"],
        },
    )


def _valid_answer():
    return {
        "hypnogram_rle": [{"start_epoch": 0, "end_epoch": 9, "stage": "W"}],
        "architecture": {"REM_pct": 0.0, "N3_pct": 0.0, "sleep_efficiency": 0.0},
    }


def test_runner_repairs_invalid_action_then_accepts_valid_answer():
    agent = _SequenceAgent([
        AgentAction(action_type=ActionType.INVALID, thought="still working"),
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer=_valid_answer()),
    ])
    result, trajectory = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=5, max_protocol_retries=2),
    )
    assert result.status == "completed"
    assert result.invalid is False
    assert result.prediction == _valid_answer()
    assert result.usage["protocol_retries"] == 1
    assert any("valid action" in m.get("content", "") for m in agent.seen_messages[1])


def test_runner_repairs_schema_invalid_final_answer():
    agent = _SequenceAgent([
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer={"record_id": "EEG_1"}),
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer=_valid_answer()),
    ])
    result, trajectory = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=5, max_protocol_retries=2),
    )
    assert result.status == "completed"
    assert result.prediction == _valid_answer()
    assert result.usage["protocol_retries"] == 1
    assert any("required field missing" in m.get("content", "") for m in agent.seen_messages[1])


def test_runner_repairs_empty_sleep_rle():
    empty_rle = {
        "hypnogram_rle": [],
        "architecture": {"REM_pct": 0.0, "N3_pct": 0.0, "sleep_efficiency": 0.0},
    }
    agent = _SequenceAgent([
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer=empty_rle),
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer=_valid_answer()),
    ])
    result, _ = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=5, max_protocol_retries=2),
    )
    assert result.status == "completed"
    assert result.usage["protocol_retries"] == 1
    assert any("at least one segment" in m.get("content", "") for m in agent.seen_messages[1])


def test_protocol_recovery_exhaustion_is_completed_format_failure():
    agent = _SequenceAgent([
        AgentAction(action_type=ActionType.INVALID, thought="no action"),
        AgentAction(action_type=ActionType.INVALID, thought="still no action"),
    ])
    result, trajectory = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=5, max_protocol_retries=1),
    )
    assert result.status == "completed"
    assert result.invalid is False
    assert result.prediction == {}
    assert result.scores == []
    assert result.usage["format_failure"] is True
    assert "protocol recovery exhausted" in (result.error or "")


def test_tool_budget_allows_final_answer_after_last_tool():
    agent = _SequenceAgent([
        AgentAction(
            action_type=ActionType.TOOL_CALL,
            tool_calls=[ToolCallRequest("call-1", "inspect", {})],
        ),
        AgentAction(action_type=ActionType.FINAL_ANSWER, final_answer=_valid_answer()),
    ])
    result, _ = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=3, max_tool_calls=1),
    )
    assert result.status == "completed"
    assert result.invalid is False
    assert result.prediction == _valid_answer()
    assert any("budget is exhausted" in m.get("content", "") for m in agent.seen_messages[1])


def test_max_turns_without_final_answer_is_completed_empty_prediction():
    agent = _SequenceAgent([
        AgentAction(action_type=ActionType.INVALID, thought="still working"),
    ])
    result, _ = run_scenario(
        scenario=_sleep_scenario(),
        agent=agent,
        env=NullToolEnvironment(),
        tool_registry=ToolRegistry(),
        evaluators=[_NoopEvaluator()],
        initial_messages=[{"role": "user", "content": "stage it"}],
        rollout_config=RolloutConfig(max_turns=1, max_protocol_retries=3),
    )
    assert result.status == "completed"
    assert result.invalid is False
    assert result.prediction == {}
    assert result.scores == []
    assert result.usage["format_failure"] is True
