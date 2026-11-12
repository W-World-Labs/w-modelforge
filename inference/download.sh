#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 HF_REPOSITORY OUTPUT_DIRECTORY [REVISION]" >&2
  exit 2
fi
hf download "$1" --local-dir "$2" --revision "${3:-main}"
python - "$2" <<'PY'
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
