#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPANDED_HF_DIRECTORY [vllm serve options]" >&2
  exit 2
fi
model="$1"
shift
if [[ ! -f "$model/w-modelforge-export.json" ]]; then
  echo "Run export_hf.py first. vLLM cannot directly load opgraph-packed-1." >&2
  exit 2
fi
exec vllm serve "$model" --dtype "${DTYPE:-bfloat16}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --served-model-name "${SERVED_MODEL_NAME:-w-modelforge}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --limit-mm-per-prompt '{"image":1,"video":0}' "$@"
