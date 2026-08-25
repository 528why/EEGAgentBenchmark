#!/usr/bin/env bash
set -euo pipefail

AGENT=${1:-configs/agents/openai_compatible.yaml}
python scripts/build_smoke.py

for task in T1 T2 T3 T4 T5 T6; do
  eeg-bench run \
    --scenario "benchmark/smoke/scenarios/${task}.jsonl" \
    --answer-key "benchmark/smoke/answer_keys/${task}.jsonl" \
    --runtime-manifest "benchmark/smoke/runtime/${task}.jsonl" \
    --agent "$AGENT" \
    --output "outputs/smoke/${task}" \
    --non-official
done
