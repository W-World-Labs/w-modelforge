"""Static public-source checks; never import model code or download assets.

Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
"""

import json
from pathlib import Path
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True)


# Original fixed Gaussian EMA tables, byte-identical to upstream 4716ffee.
# These are not per-model weights. No other tensor files are distributed.
CODEBOOKS = {
    "methods/lookatmyquant/cache/codes/trellis_L12_k1_T256_s0_ema.pt": "e53573a229b0646d22643f9f74b5ddd1c7ccbcb8",
    "methods/lookatmyquant/cache/codes/trellis_L12_k2_T256_s0_ema.pt": "0bb3e7599a098371a56daccf0bcf77deca727c2e",
    "methods/lookatmyquant/cache/codes/trellis_L12_k3_T256_s0_ema.pt": "f166c6db458a800b4bba95d21db4a71987e2e0ab",
    "methods/lookatmyquant/cache/codes/trellis_L12_k4_T256_s0_ema.pt": "54696dec2b2129a6c176d2ad1ec24ee1d565e961",
}
ASSET_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".bin", ".onnx",
    ".h5", ".hdf5", ".tflite", ".npy", ".npz", ".parquet", ".arrow",
    ".zip", ".gz", ".whl", ".pyc", ".so", ".dylib", ".dll", ".exe",
}
PRIVATE_PARTS = {".venv", ".venv-gguf", "node_modules", ".cache", "runs"}

tree_bytes = 0
python_count = json_count = shell_count = 0
paths = set()
for entry in git("ls-tree", "-rlz", "HEAD").split("\0"):
    if not entry:
        continue
    metadata, name = entry.split("\t", 1)
    mode, kind, oid, size = metadata.split()
    if kind != "blob" or mode not in {"100644", "100755"}:
        raise SystemExit(f"Unreviewed source entry: {name} ({mode} {kind})")
    path = Path(name)
    paths.add(name)
    tree_bytes += int(size)
    if name in CODEBOOKS:
        if oid != CODEBOOKS[name]:
            raise SystemExit(f"Changed upstream codebook: {name}")
    elif path.suffix.lower() in ASSET_SUFFIXES:
        raise SystemExit(f"Model, data cache or binary artifact is outside source scope: {name}")
    if (PRIVATE_PARTS.intersection(path.parts) or path.parts[0] in
            {"models", "weights", "datasets", "expanded", "gguf", "llama.cpp"} or
            path.name == ".gitmodules" or path.name.startswith((".env", ".colab-state")) or
            ("rclone" in path.name.lower() and path.suffix == ".conf")):
        raise SystemExit(f"Private state or unreviewed dependency is outside source scope: {name}")
    if path.suffix == ".py":
        compile(path.read_bytes(), name, "exec", dont_inherit=True)
        python_count += 1
    elif path.suffix == ".json":
        json.loads(path.read_text())
        json_count += 1
    elif path.suffix == ".sh":
        subprocess.run(["bash", "-n", "--", name], check=True)
        shell_count += 1

if not CODEBOOKS.keys() <= paths:
    raise SystemExit("Missing inherited codebooks; review the method source scope")
if tree_bytes > 100 * 1024 * 1024:
    raise SystemExit(f"Source tree exceeds 100 MiB: {tree_bytes} bytes")

# Inspect every reachable blob, including deleted files in the complete history.
objects = git("rev-list", "--objects", "--all", "--no-object-names")
sizes = subprocess.check_output(
    ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize)"],
    input=objects, text=True,
)
largest = 0
for line in sizes.splitlines():
    kind, oid, size = line.split()
    if kind == "blob":
        largest = max(largest, int(size))
        if int(size) > 10 * 1024 * 1024:
            raise SystemExit(f"Public-history blob exceeds 10 MiB: {oid} ({size} bytes)")

print(f"Source tree: {tree_bytes} bytes; largest history blob: {largest} bytes")
print(f"Syntax: {python_count} Python, {json_count} JSON, {shell_count} shell files")
print("Source checks do not establish model quality, GPU performance or commercial rights.")
