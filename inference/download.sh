#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: MODEL_FORMAT=packed|gguf $0 HF_REPOSITORY OUTPUT_DIRECTORY [REVISION]" >&2
  exit 2
fi
format="${MODEL_FORMAT:-packed}"
case "$format" in
  packed) hf download "$1" --local-dir "$2" --revision "${3:-main}" --exclude 'gguf-adaptive/*' ;;
  gguf) hf download "$1" --local-dir "$2" --revision "${3:-main}" --include 'gguf-adaptive/*' ;;
  *) echo "MODEL_FORMAT must be packed or gguf" >&2; exit 2 ;;
esac
python - "$2" "$format" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def verify(path, expected_size, expected_hash):
    if path.stat().st_size != expected_size:
        raise SystemExit(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"SHA256 mismatch: {path}")

if sys.argv[2] == "gguf":
    root = root / "gguf-adaptive"
    checks = json.loads((root / "checksums.json").read_text())
    if not checks["files"]:
        raise SystemExit("Empty GGUF checksums")
    for name, item in checks["files"].items():
        if Path(name).name != name:
            raise SystemExit("Invalid GGUF filename in checksums")
        verify(root / name, item["bytes"], item["sha256"])
    print("Downloaded native adaptive GGUF package; listed files SHA256 verified.")
    raise SystemExit(0)

checks = json.loads((root / "checksums.json").read_text())
for name, item in checks["files"].items():
    if Path(name).name != name:
        raise SystemExit("Invalid packed filename in checksums")
    verify(root / "packed" / name, item["bytes"], item["sha256"])
vision = json.loads((root / "vision-provenance.json").read_text())
if vision["weights_file"] != "vision.safetensors":
    raise SystemExit("Unexpected original vision filename")
verify(root / vision["weights_file"], vision["weights_bytes"], vision["weights_sha256"])
print("Downloaded release; packed files and original BF16 vision SHA256 verified.")
PY
