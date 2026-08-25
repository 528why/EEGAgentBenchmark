from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_agent_bench.core.scenario import load_scenarios
from eeg_agent_bench.envs.tool_env import ToolEnvironment
from eeg_agent_bench.tasks.registry import get_task_bundle
from eeg_agent_bench.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


class _ForbiddenGold(dict):
    """Fail if prompt or tool setup attempts to inspect private gold."""

    def _fail(self, *args, **kwargs):
        raise AssertionError("gold_private was accessed outside scoring")

    __bool__ = _fail
    __contains__ = _fail
    __getitem__ = _fail
    __iter__ = _fail
    __len__ = _fail
    get = _fail
    items = _fail
    keys = _fail
    values = _fail


@pytest.mark.parametrize("task_id", [f"T{i}" for i in range(1, 7)])
def test_gold_is_inaccessible_to_prompts_and_tool_setup(task_id: str):
    scenario = load_scenarios(
        ROOT / "benchmark/scenarios" / f"{task_id}.jsonl",
        answer_key_path=ROOT / "benchmark/answer_keys" / f"{task_id}.jsonl",
        runtime_manifest_path=ROOT / "benchmark/runtime" / f"{task_id}.jsonl",
    )[0]
    scenario.gold_private = _ForbiddenGold()

    messages = get_task_bundle(task_id).task.build_messages(scenario, [])
    prompt = json.dumps(messages, ensure_ascii=False)
    assert "gold_private" not in prompt

    environment = ToolEnvironment(ToolRegistry())
    assert environment.reset(scenario) == []
    assert "gold_private" not in environment._context


def test_runtime_paths_are_merged_in_memory_but_not_exposed_to_prompt():
    scenarios = load_scenarios(
        ROOT / "benchmark/scenarios/T3.jsonl",
        answer_key_path=ROOT / "benchmark/answer_keys/T3.jsonl",
        runtime_manifest_path=ROOT / "benchmark/runtime/T3.jsonl",
    )
    scenario = scenarios[0]
    runtime_path = Path(scenario.records[0].data_path)
    assert runtime_path.is_absolute()
    assert runtime_path.parts[-4:-1] == ("data", "bonn", "Z")
    assert scenario.gold_private["label"] in {"normal", "epileptic"}

    messages = get_task_bundle("T3").task.build_messages(scenario, [])
    prompt = json.dumps(messages, ensure_ascii=False)
    assert str(runtime_path) not in prompt
    assert "gold_private" not in prompt
    assert "set_letter" not in prompt
    assert "label_binary" not in prompt


def test_public_scenarios_have_no_gold_or_runtime_paths():
    for index in range(1, 7):
        path = ROOT / "benchmark/scenarios" / f"T{index}.jsonl"
        text = path.read_text(encoding="utf-8")
        assert "gold_private" not in text
        assert '"data_path"' not in text
        assert "/mnt/" not in text


def test_official_signal_tasks_offer_exactly_ten_tools():
    expected = set(
        json.loads((ROOT / "benchmark/benchmark.json").read_text())["official_tools"]
    )
    assert len(expected) == 10
    for index in range(2, 7):
        first = json.loads(
            (ROOT / "benchmark/scenarios" / f"T{index}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert set(first["access"]["allowed_tools"]) == expected


def test_all_tasks_use_1000_turn_limit():
    benchmark = json.loads((ROOT / "benchmark/benchmark.json").read_text())
    assert {task["max_turns"] for task in benchmark["tasks"].values()} == {1000}

    for scenario_root in ("scenarios", "smoke/scenarios"):
        for index in range(1, 7):
            rows = (
                ROOT / "benchmark" / scenario_root / f"T{index}.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            assert rows
            assert {
                json.loads(row)["access"]["max_turns"] for row in rows if row.strip()
            } == {1000}
