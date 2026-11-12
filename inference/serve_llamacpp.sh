#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LANGUAGE_GGUF MMPROJ_GGUF [llama-server options]" >&2
  exit 2
fi
model="$1"
projector="$2"
shift 2
if [[ ! -f "$model" || ! -f "$projector" ]]; then
  echo "Both the converted language GGUF and unquantized mmproj GGUF are required." >&2
  exit 2
fi
exec "${LLAMA_SERVER:-llama-server}" -m "$model" --mmproj "$projector" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --alias "${SERVED_MODEL_NAME:-w-modelforge}" \
  --ctx-size "${MAX_MODEL_LEN:-4096}" --n-gpu-layers "${N_GPU_LAYERS:-99}" --jinja "$@"
