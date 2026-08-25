#!/usr/bin/env bash
set -euo pipefail

AGENT=${1:-configs/agents/openai_compatible.yaml}
RUN_NAME=${2:-$(basename "${AGENT%.yaml}")}

exec eeg-bench run-all --agent "$AGENT" --run-name "$RUN_NAME"
