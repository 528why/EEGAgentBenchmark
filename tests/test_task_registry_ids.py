from eeg_agent_bench.tasks.registry import (
    display_task_id,
    get_task_bundle,
    get_task_meta,
    list_task_meta,
    resolve_task_key,
)


def test_paper_order_and_regimes():
    metas = list_task_meta()
    assert [meta.display_id for meta in metas] == [f"T{i}" for i in range(1, 7)]
    assert [meta.regime for meta in metas] == [
        "knowledge_reasoning",
        "short_horizon_analysis",
        "short_horizon_analysis",
        "short_horizon_analysis",
        "long_horizon_analysis",
        "long_horizon_analysis",
    ]


def test_semantic_and_legacy_ids_resolve_to_same_task():
    cases = {
        "knowledge_qa": ("T1", "C6", "K0"),
        "artifact_identification": ("T2", "C2", "C2-Artifact"),
        "epilepsy_screening": ("T3", "C3", "C3-Routine"),
        "dementia_cohort_classification": ("T4", "C4", "C3-Cohort"),
        "seizure_detection": ("T5", "C1", "C1-Seizure"),
        "sleep_staging": ("T6", "C5", "C3-Sleep"),
    }
    for task_key, identifiers in cases.items():
        expected_bundle = get_task_bundle(task_key)
        for identifier in identifiers:
            assert resolve_task_key(identifier) == task_key
            assert get_task_bundle(identifier) is expected_bundle
            assert get_task_meta(identifier).task_key == task_key
            assert display_task_id(identifier) == identifiers[0]


def test_t1_is_unambiguously_knowledge_qa():
    assert resolve_task_key("T1") == "knowledge_qa"
    assert get_task_meta("T1").display_name_en == "EEG Knowledge QA"
