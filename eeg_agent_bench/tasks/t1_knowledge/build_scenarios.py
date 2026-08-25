"""Build T1 (EEG Knowledge QA) scenarios from the strict MedMCQA-EEG subset.

Input  : the strict-filtered MedMCQA records produced by
         ``scripts/filter_medmcqa_eeg_strict.py`` (each record carries
         question, opa..opd, cop in {0,1,2,3}, choice_type=="single").
Output : :class:`Scenario` objects with ``task_id="T1"``, no tools.

The correct option index ``cop`` (0->A .. 3->D) becomes the private gold
letter.  Options are rendered visibly to the agent; the answer letter is
the only gold.
"""

from __future__ import annotations

import logging
from typing import Any

from eeg_agent_bench.tasks.t1_knowledge.schema import T1_MCQ_OUTPUT_SCHEMA
from eeg_agent_bench.types import (
    AccessConfig,
    AccessMode,
    EvaluatorConfig,
    ReleaseConfig,
    Scenario,
)

logger = logging.getLogger(__name__)

_LETTERS = ("A", "B", "C", "D")
_TASK_ID = "T1"
_TASK_NAME = "eeg_knowledge_qa"


def _t1_access() -> AccessConfig:
    # Knowledge-only: no tools.  2 turns is plenty (1 to answer, 1 slack).
    return AccessConfig(
        mode=AccessMode.METADATA_ONLY_WITHOUT_TOOLS,
        allowed_tools=[],
        max_turns=1000,
    )


def _t1_evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(primary="mcq_correct", secondary=[])


def _shuffle_options(
    options: dict[str, str], gold_letter: str, seed_key: str
) -> tuple[dict[str, str], str]:
    """Deterministically permute the 4 options to remove answer-position bias.

    MedMCQA's native answer skews toward option A; a model that always answers
    A scores far above chance.  We re-assign option *texts* to letters using a
    per-record deterministic RNG (seeded from the record id, so the bench stays
    reproducible) and remap the gold letter to wherever the correct text lands.
    """
    import hashlib
    import random

    gold_text = options[gold_letter]
    texts = [options[L] for L in _LETTERS]
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(texts)
    new_options = {L: t for L, t in zip(_LETTERS, texts)}
    new_gold = next(L for L, t in new_options.items() if t == gold_text)
    return new_options, new_gold


def build_t1_scenario(rec: dict[str, Any]) -> Scenario:
    """Build a single T1 scenario from one strict MedMCQA record."""
    options = {letter: str(rec[f"op{letter.lower()}"]).strip() for letter in _LETTERS}
    cop = int(rec["cop"])
    gold_letter = _LETTERS[cop]

    rec_id = str(rec.get("id", ""))
    # Remove answer-position bias (native MedMCQA skews to A).
    options, gold_letter = _shuffle_options(options, gold_letter, rec_id)
    return Scenario(
        scenario_id=f"k0_medmcqa_{rec_id}",
        task_id=_TASK_ID,
        task_name=_TASK_NAME,
        dataset="MedMCQA-EEG-strict",
        dataset_version="2026-06-22",
        records=[],
        input={
            "question": str(rec["question"]).strip(),
            "options": options,
            # provenance (not used for scoring, useful for analysis / strata)
            "subject_name": rec.get("subject_name"),
            "eeg_strict_categories": rec.get("eeg_strict_categories", []),
            "orig_split": rec.get("_orig_split"),
        },
        access=_t1_access(),
        expected_output_schema=T1_MCQ_OUTPUT_SCHEMA,
        gold_private={
            "answer": gold_letter,
            "answer_text": options[gold_letter],
        },
        evaluator=_t1_evaluator(),
        release=ReleaseConfig(split="candidate", include_gold_in_release=False),
    )


def build_t1_scenarios(records: list[dict[str, Any]]) -> list[Scenario]:
    """Build T1 scenarios for all strict MedMCQA records."""
    scenarios: list[Scenario] = []
    for rec in records:
        try:
            scenarios.append(build_t1_scenario(rec))
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Skipping malformed T1 record {rec.get('id')}: {e}")
    return scenarios


__all__ = ["build_t1_scenario", "build_t1_scenarios"]
