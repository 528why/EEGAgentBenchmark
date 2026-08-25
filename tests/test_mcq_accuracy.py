"""Unit tests for the K0 MCQ accuracy evaluator + answer normalisation."""

from eeg_agent_bench.evaluators.mcq_accuracy import (
    MCQAccuracyEvaluator,
    normalise_answer,
)
from eeg_agent_bench.tasks.t1_knowledge.build_scenarios import build_t1_scenario

_OPTS = {"A": "8-13", "B": "4-7", "C": "13-30", "D": "0-4"}


def test_normalise_letter_forms():
    assert normalise_answer("B", _OPTS) == "B"
    assert normalise_answer("(C)", _OPTS) == "C"
    assert normalise_answer("D.", _OPTS) == "D"
    assert normalise_answer("Option B", _OPTS) == "B"
    assert normalise_answer("answer: A", _OPTS) == "A"
    assert normalise_answer("the answer is C", _OPTS) == "C"


def test_normalise_option_text():
    assert normalise_answer("8-13", _OPTS) == "A"


def test_normalise_unrecoverable():
    assert normalise_answer("Delta", _OPTS) == ""
    assert normalise_answer("", _OPTS) == ""
    assert normalise_answer(None, _OPTS) == ""


def _record(cop):
    return {
        "id": "x1",
        "question": "Alpha rhythm frequency (Hz)?",
        "opa": "0-4", "opb": "4-7", "opc": "8-13", "opd": "13-30",
        "cop": cop, "choice_type": "single",
    }


def test_evaluator_correct_and_wrong():
    scen = build_t1_scenario(_record(2))
    ev = MCQAccuracyEvaluator()
    gold = scen.gold_private["answer"]
    wrong = next(letter for letter in "ABCD" if letter != gold)
    assert ev.evaluate(scen, {"answer": gold}).value == 1.0
    assert ev.evaluate(scen, {"answer": wrong}).value == 0.0
    # option-text reply also graded
    assert ev.evaluate(scen, {"answer": "8-13"}).value == 1.0
    # empty / missing prediction -> 0
    assert ev.evaluate(scen, {}).value == 0.0


def test_scenario_no_label_leak():
    scen = build_t1_scenario(_record(1))
    # gold answer must live ONLY in gold_private, never in input
    assert scen.gold_private["answer"] in "ABCD"
    assert scen.input["options"][scen.gold_private["answer"]] == "4-7"
    assert "answer" not in scen.input
    assert scen.access.allowed_tools == []
