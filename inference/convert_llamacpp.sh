#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 EXPANDED_HF_DIRECTORY LLAMA_CPP_CHECKOUT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
model="$1"
engine="$2"
output="$3"
if [[ ! -f "$model/w-modelforge-export.json" || ! -f "$engine/convert_hf_to_gguf.py" ]]; then
  echo "Run export_hf.py first and provide a llama.cpp checkout; see inference/README.md." >&2
  exit 2
fi
language_type="$(python -c 'import json,sys; print({"float32":"f32","bfloat16":"bf16"}[json.load(open(sys.argv[1]))["language_dtype"]])' "$model/w-modelforge-export.json")"
if [[ -e "$output" ]]; then
  echo "Output directory already exists; choose a new directory." >&2
  exit 2
fi
mkdir -p "$output"
python "$engine/convert_hf_to_gguf.py" "$model" --no-mtp \
  --outtype "$language_type" --outfile "$output/language.gguf"
python "$engine/convert_hf_to_gguf.py" "$model" --mmproj \
  --outtype bf16 --outfile "$output/mmproj.gguf"
engine_revision="$(git -C "$engine" rev-parse HEAD)"
python - "$output" "$language_type" "$engine_revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {}
for name in ("language.gguf", "mmproj.gguf"):
    path = root / name
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    files[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
(root / "conversion.json").write_text(json.dumps({
    "llama_cpp_commit": sys.argv[3], "language_outtype": sys.argv[2],
    "mmproj_outtype": "bf16 with converter-required F32 tensors", "mtp": False,
    "files": files, "warning": "Layout conversion; backend quality has not been established by this script."
}, indent=2) + "\n")
print("GGUF files and conversion receipt written; no additional low-bit quantization applied.")
PY
