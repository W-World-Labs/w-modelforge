"""W-ModelForge LookAtMyQuant text PTQ pilot on a Colab G4.

Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_REPO = "https://github.com/W-World-Labs/w-modelforge.git"
COMMIT = "a5308ddbaa3b7577e4bb380c688e86c1971c70f7"
UPSTREAM_COMMIT = "4716ffee3da205a8d1df94e4abc629040893fa53"
MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ROOT = Path("/content/model-forge")
CHECKOUT = ROOT / "source"
SOURCE = CHECKOUT / "methods/lookatmyquant"
OUTPUT = ROOT / "output"
WORK = ROOT / "work"
REMOTE = ""
DTYPE = "float32"
CPU_OFFLOAD = True


def run(*command, cwd=None):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=SOURCE if cwd is None else cwd, check=True)


def python(script, *arguments):
    run(sys.executable, "-u", script, *arguments)


def save(name, value):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sync():
    if (WORK / "run.log").exists():
        shutil.copyfile(WORK / "run.log", OUTPUT / "run.log")
    run("rclone", "copy", OUTPUT, REMOTE, "--transfers", "2", "--checkers", "4",
        "--retries", "5", "--stats", "30s", cwd=ROOT)
    run("rclone", "check", OUTPUT, REMOTE, "--one-way", cwd=ROOT)


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    if not CHECKOUT.exists():
        run("git", "clone", SOURCE_REPO, CHECKOUT, cwd=ROOT)
    run("git", "checkout", "--detach", COMMIT, cwd=CHECKOUT)
    code_notices = OUTPUT / "code-source"
    code_notices.mkdir(exist_ok=True)
    for name in ("LICENSE", "NOTICE.md", "UPSTREAM.md"):
        shutil.copyfile(CHECKOUT / name, code_notices / name)
    # Keep Colab's preinstalled Blackwell-compatible PyTorch build; record its exact version below.
    requirements = [line for line in (SOURCE / "requirements.txt").read_text().splitlines()
                    if not line.startswith("torch==")]
    (WORK / "requirements.txt").write_text("\n".join(requirements) + "\n")
    run(sys.executable, "-m", "pip", "install", "-r", WORK / "requirements.txt")
    if not shutil.which("rclone"):
        run("apt-get", "update", cwd=ROOT)
        run("apt-get", "install", "-y", "rclone", cwd=ROOT)
    # Upload only the selected rclone remote's configuration before running this script.
    run("rclone", "lsd", REMOTE.split(":", 1)[0] + ":", "--max-depth", "1", cwd=ROOT)
    import torch
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 88 * 1024**3:
        raise RuntimeError(f"These pilots require the 96 GB G4; got {gpu.name}.")
    save("environment.json", {"model": MODEL, "source_repo": SOURCE_REPO, "source_commit": COMMIT,
                              "upstream_commit": UPSTREAM_COMMIT, "gpu": gpu.name,
                              "vram_bytes": gpu.total_memory, "torch": torch.__version__,
                              "dtype": DTYPE, "cpu_offload": CPU_OFFLOAD, "scope": "text-only",
                              "source_method": "methods/lookatmyquant",
                              "recipe": "PTQ pilot; no distillation; linears target 2.7 bpw; tables lm4",
                              "source": SOURCE_REPO})
    (OUTPUT / "requirements-resolved.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    from huggingface_hub import hf_hub_download, snapshot_download
    model_dir = snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE*", "NOTICE*", "README.md"],
                                  ignore_patterns=["original/*", "mtp/*"])
    model_notices = OUTPUT / "model-source"
    model_notices.mkdir(exist_ok=True)
    license_assets = []
    for name in ("LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt", "README.md"):
        if (Path(model_dir) / name).is_file():
            shutil.copyfile(Path(model_dir) / name, model_notices / name)
            license_assets.append(name)
    (model_notices / "MODIFICATIONS.md").write_text(
        f"Modified model weights: W-World-Labs text-only PTQ with LookAtMyQuant; "
        f"trellis/LDLQ/Hadamard, Fisher row allocation, linears target 2.7 bpw, "
        f"vocabulary tables lm4, no distillation. Source: {MODEL} at {MODEL_REVISION}. "
        f"Model license: Apache-2.0; compiler code is separately MIT. "
        f"Custom packed format; final size/quality require completed evaluation.\n")
    # An upstream import reads this small config even for models using the generic harness.
    hf_hub_download("HuggingFaceTB/SmolLM2-360M", "config.json")
    from datasets import load_dataset
    text_dir = Path("/root/.cache/huggingface/compiler_text")
    text_dir.mkdir(parents=True, exist_ok=True)
    sources = {"en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
               "zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", "text"),
               "code": ("codeparrot/codeparrot-clean", None, "content")}
    for name, (repo, config, field) in sources.items():
        path = text_dir / f"{name}.json"
        if path.exists():
            continue
        docs, count = [], 0
        for row in load_dataset(repo, config, split="train", streaming=True):
            text = row[field]
            if not text or len(text) < 200:
                continue
            docs.append(text[:12000])
            count += len(docs[-1])
            if count >= 500000:
                break
        path.write_text(json.dumps({"repo": repo, "config": config, "split": "train", "docs": docs}))
    math_sources = WORK / "math-sources"
    snapshot_download("EleutherAI/hendrycks_math", repo_type="dataset", cache_dir=math_sources,
                      allow_patterns=["*/train-*.parquet"])
    save("source-model.json", {"repo": MODEL, "snapshot": Path(model_dir).name,
                              "path": model_dir, "math_sources": str(math_sources),
                              "license": "Apache-2.0", "license_assets": license_assets})
    return model_dir, math_sources


def allocate(model_dir):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    sys.path.insert(0, str(SOURCE))
    import compiler_arch as arch
    from rowgroup_alloc import allocate as solve, make_row_groups
    importance = torch.load(WORK / "importance.pt", map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir))
    prefix, layers = arch.decoder_layers(model)
    matrices, linears = {}, {}
    for i, layer in enumerate(layers):
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear) or min(module.weight.shape) < 64:
                continue
            short = f"{i}.{name}"
            full = f"{prefix}.{short}.weight"
            linears[short] = "tr3x"
            matrices[full] = {"importance": importance[short].tolist(),
                              "columns": module.in_features, "base_bits": 3}
    groups = make_row_groups(matrices, {b: 4.0**-b for b in (1, 2, 3, 4)})
    weights = sum(len(m["importance"]) * m["columns"] for m in matrices.values())
    # Payload budget only. Whole-model bytes are measured after packing, never inferred from bpw.
    minimum_overhead = sum(g.options[0].payload_bytes - len(g.rows) * g.columns * 2 // 8 for g in groups)
    budget = int(weights * 2.7 / 8) + minimum_overhead
    row_plan = solve(groups, budget).as_dict()
    row_plan["distortion_provenance"] = "Fisher-weighted 4^-bits proxy; quality measured separately"
    save("row-plan.json", row_plan)
    # 9B has an untied head. Both large vocabulary tables must be compressed and counted.
    table = {"sizes": [arch.text_config(model.config).vocab_size], "programs": ["lm4"],
             "order": "frequency in the text mix"}
    save("plan.json", {"model": MODEL, "linears": linears, "embed": table,
                       "head": table, "trellis fit": "ema"})


def main():
    global MODEL, MODEL_REVISION, ROOT, CHECKOUT, SOURCE, OUTPUT, WORK, REMOTE, DTYPE, CPU_OFFLOAD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"], default=MODEL)
    parser.add_argument("--drive-remote", default=os.environ.get("MODEL_FORGE_DRIVE_REMOTE"),
                        help="existing rclone remote name, without colon")
    parser.add_argument("--drive-directory", default="model-quantization")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d"))
    parser.add_argument("--calibration-batch", type=int, default=None)
    parser.add_argument("--stage", choices=["setup", "calibrate", "allocate", "build", "pack", "evaluate", "sync", "all"], default="all")
    args = parser.parse_args()
    if not args.drive_remote or ":" in args.drive_remote:
        parser.error("set --drive-remote or MODEL_FORGE_DRIVE_REMOTE to an existing rclone remote name")
    if not args.run_id or "/" in args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id must be one non-empty directory name")
    if args.calibration_batch is not None and args.calibration_batch < 1:
        parser.error("--calibration-batch must be positive")
    is_27b = args.model == "Qwen/Qwen3.8-27B"
    if is_27b:
        MODEL = args.model
        MODEL_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
        DTYPE, CPU_OFFLOAD = "bfloat16", True
    ROOT = Path("/content/model-forge") / args.run_id / MODEL.split("/")[-1]
    CHECKOUT = ROOT / "source"
    SOURCE, OUTPUT, WORK = CHECKOUT / "methods/lookatmyquant", ROOT / "output", ROOT / "work"
    REMOTE = f"{args.drive_remote}:{args.drive_directory.strip('/')}/{MODEL.split('/')[-1]}-LAMQ2-{args.run_id}"
    calibration_batch = args.calibration_batch or (8 if is_27b else 4)
    dtype_args = ["--dtype", DTYPE] if CPU_OFFLOAD else []
    os.environ.update(OPGRAPH_DEVICE="cuda", OPGRAPH_THREADS="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    if args.stage == "sync":
        sync()
        return
    stages = ["setup", "calibrate", "allocate", "build", "pack", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        marker = OUTPUT / f"{stage}.done.json"
        if marker.exists():
            print("Already completed:", stage, flush=True)
            sync()  # Retry backup even when this stage's computation already finished.
            continue
        started = time.time()
        save("status.json", {"stage": stage, "state": "running", "started_utc": started})
        try:
            if stage == "setup":
                setup()
            else:
                source = json.loads((OUTPUT / "source-model.json").read_text())
                model_dir = source["path"]
                if stage == "calibrate":
                    python("math_calibration.py", "--model-path", model_dir,
                           "--math-sources", source["math_sources"], "--problems", "128",
                           "--max-new", "1024", "--len", "1024", "--batch", calibration_batch,
                           *dtype_args, "--out", WORK / "math.pt")
                    python("row_importance.py", "--model-path", model_dir, "--windows", WORK / "math.pt",
                           "--n", "12", "--len", "512", *dtype_args,
                           "--save", WORK / "importance.pt", "--out", OUTPUT / "importance.json")
                elif stage == "allocate":
                    allocate(model_dir)
                elif stage == "build":
                    if is_27b and shutil.disk_usage(ROOT).free < 130_000_000_000:
                        raise RuntimeError("27B build needs 130 GB free for dense weights, codes and packed output; "
                                           "finish 9B evaluation and verified backup before removing its dense.pt/codes.pt.")
                    python("quantize.py", "--program", "plan", "--model", MODEL, "--model-path", model_dir,
                           "--plan", OUTPUT / "plan.json", "--row-plan", OUTPUT / "row-plan.json",
                           "--dtype", DTYPE, *(["--cpu-offload"] if CPU_OFFLOAD else []),
                           "--rot", "in-shared", "--trellis-rows", "16,8",
                           "--trellis-fit", "ema", "--seeds", "1", "--seed-set", "1",
                           "--data", "mix", "--calib", "32", "--calib-len", "1024", "--chunk", "1",
                           "--calib-math", "1.0", "--math-cache", WORK / "math.pt",
                           "--token-weights", "digits", "--token-floor", "0.1", "--cross-alpha", "0.5",
                           "--ctx", "512", "--eval", "8", "--eval-mix", "2",
                           "--save", WORK / "dense.pt", "--save-codes", WORK / "codes.pt", "--out", OUTPUT / "quality.json")
                elif stage == "pack":
                    if CPU_OFFLOAD:
                        # Uncoded values retain the build dtype; 9B stays FP32, while 27B avoids an FP32 source copy.
                        run(sys.executable, "-c", "from row_allocation import load_original_model, pack_codes; "
                            "import sys; m=load_original_model(sys.argv[1], dtype=sys.argv[4], device='cpu'); "
                            "print(pack_codes(sys.argv[2], sys.argv[1], sys.argv[3], original_model=m))",
                            model_dir, WORK / "codes.pt", OUTPUT / "packed", DTYPE)
                    else:
                        python("row_allocation.py", "pack", "--source-dir", SOURCE, "--model-path", model_dir,
                               "--trial-dir", WORK, "--codes", WORK / "codes.pt", "--output", OUTPUT / "packed")
                elif stage == "evaluate":
                    python("math_eval.py", "--model-path", model_dir, "--models", f"orig,dense:{WORK / 'dense.pt'}",
                           "--problems-file", SOURCE / "results/protocols/math216.json", "--no-thinking",
                           "--batch", 8 if is_27b else 4, "--max-new", "1024", *dtype_args,
                           *(["--cache-tag", MODEL.split("/")[-1].lower()] if CPU_OFFLOAD else []),
                           "--out", OUTPUT / "math216.json")
            save(marker.name, {"seconds": round(time.time() - started)})
            save("status.json", {"stage": stage, "state": "completed", "seconds": round(time.time() - started)})
            sync()
        except Exception as error:
            save("status.json", {"stage": stage, "state": "failed", "error": str(error)})
            try:
                sync()
            except Exception as sync_error:
                print("Backup also failed:", sync_error, flush=True)
            raise
    if args.stage == "all":
        save("status.json", {"state": "complete", "recipe": "PTQ pilot, no distillation"})
        sync()


if __name__ == "__main__":
    main()
