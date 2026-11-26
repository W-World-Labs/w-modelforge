"""Static public-source checks; never import model code or download assets.

Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
"""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "upstream/lookatmyquant"
UPSTREAM_URL = "https://github.com/ShinMegamiBoson/lookatmyquant.git"
PREPARED = ROOT / "methods/lookatmyquant"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True)


# Original fixed Gaussian EMA tables, byte-identical to upstream 4716ffee.
# These are not per-model weights. No other tensor files are distributed.
CODEBOOKS = {
    "cache/codes/trellis_L12_k1_T256_s0_ema.pt": "e53573a229b0646d22643f9f74b5ddd1c7ccbcb8",
    "cache/codes/trellis_L12_k2_T256_s0_ema.pt": "0bb3e7599a098371a56daccf0bcf77deca727c2e",
    "cache/codes/trellis_L12_k3_T256_s0_ema.pt": "f166c6db458a800b4bba95d21db4a71987e2e0ab",
    "cache/codes/trellis_L12_k4_T256_s0_ema.pt": "54696dec2b2129a6c176d2ad1ec24ee1d565e961",
}
ASSET_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".bin", ".onnx",
    ".h5", ".hdf5", ".tflite", ".npy", ".npz", ".parquet", ".arrow",
    ".zip", ".gz", ".whl", ".pyc", ".so", ".dylib", ".dll", ".exe",
}
PRIVATE_PARTS = {".venv", ".venv-gguf", "node_modules", ".cache", "runs"}

def tree(cwd):
    for entry in git("ls-tree", "-rlz", "HEAD", cwd=cwd).split("\0"):
        if entry:
            metadata, name = entry.split("\t", 1)
            yield (*metadata.split(), name)


def check_path(name, codebooks, *, allow_gitmodules=False):
    path = Path(name)
    if name not in codebooks and path.suffix.lower() in ASSET_SUFFIXES:
        raise SystemExit(f"Model, data cache or binary artifact is outside source scope: {name}")
    if (PRIVATE_PARTS.intersection(path.parts) or path.parts[0] in
            {"models", "weights", "datasets", "expanded", "gguf", "llama.cpp"} or
            (path.name == ".gitmodules" and not (allow_gitmodules and name == ".gitmodules")) or
            path.name.startswith((".env", ".colab-state")) or
            ("rclone" in path.name.lower() and path.suffix == ".conf")):
        raise SystemExit(f"Private state or unreviewed dependency is outside source scope: {name}")


def check_syntax(path):
    if path.suffix == ".py":
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    elif path.suffix == ".json":
        json.loads(path.read_text())
    elif path.suffix == ".sh":
        subprocess.run(["bash", "-n", "--", path], check=True)


def check_history(cwd, ref):
    # Inspect every reachable blob, including deleted files.
    objects = git("rev-list", "--objects", ref, "--no-object-names", cwd=cwd)
    sizes = subprocess.check_output(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize)"],
        input=objects, cwd=cwd, text=True,
    )
    largest = 0
    for line in sizes.splitlines():
        kind, oid, size = line.split()
        if kind == "blob":
            largest = max(largest, int(size))
            if int(size) > 10 * 1024 * 1024:
                raise SystemExit(f"Public-history blob exceeds 10 MiB: {oid} ({size} bytes)")
    return largest


def main():
    expected_config = {
        f"submodule.{UPSTREAM}.path={UPSTREAM}",
        f"submodule.{UPSTREAM}.url={UPSTREAM_URL}",
    }
    config = (ROOT / ".gitmodules").read_bytes()
    if config != subprocess.check_output(["git", "show", "HEAD:.gitmodules"], cwd=ROOT):
        raise SystemExit("The submodule configuration differs from the checked commit")
    entries = git("config", "--blob", "HEAD:.gitmodules", "--list").splitlines()
    if len(entries) != len(expected_config) or set(entries) != expected_config:
        raise SystemExit("Only the reviewed LookAtMyQuant submodule URL is allowed")

    pin = None
    tree_bytes = 0
    syntax = {".py": 0, ".json": 0, ".sh": 0}
    for mode, kind, oid, size, name in tree(ROOT):
        if (mode, kind, name) == ("160000", "commit", UPSTREAM):
            pin = oid
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise SystemExit(f"Unreviewed source entry: {name} ({mode} {kind})")
        if name.startswith("methods/lookatmyquant/"):
            raise SystemExit(f"Tracked source duplicates the upstream dependency: {name}")
        check_path(name, {}, allow_gitmodules=True)
        tree_bytes += int(size)
        path = ROOT / name
        if path.suffix in syntax:
            check_syntax(path)
            syntax[path.suffix] += 1
    if not pin or git("rev-parse", f":{UPSTREAM}").strip() != pin:
        raise SystemExit("The LookAtMyQuant submodule does not match the checked commit")

    upstream = ROOT / UPSTREAM
    subprocess.run(
        ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
         "-c", f"submodule.{UPSTREAM}.url={UPSTREAM_URL}",
         "submodule", "update", "--init", "--checkout", "--", UPSTREAM],
        cwd=ROOT, check=True,
    )
    if git("rev-parse", "HEAD", cwd=upstream).strip() != pin:
        raise SystemExit("LookAtMyQuant is not checked out at the recorded commit")
    codebooks = {}
    upstream_bytes = 0
    for mode, kind, oid, size, name in tree(upstream):
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise SystemExit(f"Unreviewed upstream source entry: {name} ({mode} {kind})")
        check_path(name, CODEBOOKS)
        if name in CODEBOOKS:
            codebooks[name] = oid
        upstream_bytes += int(size)
    if codebooks != CODEBOOKS:
        raise SystemExit("The pinned upstream codebooks differ from the reviewed blobs")
    if tree_bytes + upstream_bytes > 100 * 1024 * 1024:
        raise SystemExit("W-ModelForge and pinned upstream source exceed 100 MiB")
    subprocess.run(["bash", "scripts/prepare_lookatmyquant.sh"], cwd=ROOT, check=True)

    prepared_bytes = 0
    for path in PREPARED.rglob("*"):
        name = path.relative_to(PREPARED).as_posix()
        if path.is_symlink():
            raise SystemExit(f"Unexpected symlink in prepared source: {name}")
        if not path.is_file():
            continue
        check_path(name, CODEBOOKS)
        if name in CODEBOOKS and git("hash-object", str(path)).strip() != CODEBOOKS[name]:
            raise SystemExit(f"The prepared patch changed an upstream codebook: {name}")
        prepared_bytes += path.stat().st_size
        if path.suffix in syntax:
            check_syntax(path)
            syntax[path.suffix] += 1
    if not all((PREPARED / name).is_file() for name in CODEBOOKS):
        raise SystemExit("Missing upstream codebooks in the prepared source")
    if tree_bytes + prepared_bytes > 100 * 1024 * 1024:
        raise SystemExit("W-ModelForge and prepared method source exceed 100 MiB")

    largest = max(check_history(ROOT, "--all"), check_history(upstream, pin))
    print(f"Source tree with prepared method: {tree_bytes + prepared_bytes} bytes; largest history blob: {largest} bytes")
    print(f"Syntax: {syntax['.py']} Python, {syntax['.json']} JSON, {syntax['.sh']} shell files")
    print("Source checks do not establish model quality, GPU performance or commercial rights.")


if __name__ == "__main__":
    main()
