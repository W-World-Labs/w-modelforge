#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RELEASE_DIRECTORY --prompt TEXT [--image FILE] [loader options]" >&2
  exit 2
fi
release="$1"
shift
repo="${W_MODELFORGE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
loader="$repo/methods/lookatmyquant/load_packed.py"
if [[ -f "$repo/scripts/prepare_lookatmyquant.sh" ]]; then
  bash "$repo/scripts/prepare_lookatmyquant.sh"
fi
if [[ ! -f "$loader" ]]; then
  echo "Set W_MODELFORGE_ROOT to your W-ModelForge checkout; see inference/README.md." >&2
  exit 2
fi
dtype="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_dtype"])' "$release/evaluation.json")"
exec python "$loader" "$release/packed" --dtype "$dtype" \
  --device "${DEVICE:-cuda}" --decode-device "${DECODE_DEVICE:-${DEVICE:-cuda}}" \
  --vision-file "$release/vision.safetensors" --processor-dir "$release" "$@"
