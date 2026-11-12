"""Incremental math216 evaluation, one original or packed GPU model per process.

--output is a dedicated directory containing original.json, packed.json and
comparison.json. Re-run the identical command to resume complete saved batches.
27B uses BF16/batch 8 with --max-new 2048 4096; 9B uses FP32/batch 4 with
--max-new 4096. Each cap generates its own answers; no answer is truncated from
another cap. Optional rclone copies/checks only those three small JSON reports.
"""

# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
# Reuses Shin Megami Boson's LookAtMyQuant generation and scoring (MIT).

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from row_allocation import _write_json, file_sha256, load_original_model

SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE / "results/protocols/math216.json"
REPORTS = ("original.json", "packed.json", "comparison.json")
COMMON_TEXT_ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
OPTIONAL_TEXT_ASSETS = ("special_tokens_map.json", "added_tokens.json",
                       "chat_template.json", "vocab.json", "merges.txt")
SOURCES = ("compare_math_caps.py", "math_eval.py", "math_bench.py", "row_allocation.py",
           "load_packed.py", "compiler_artifact.py", "opgraph/hadamard.py")
SCHEDULE = {"Qwen/Qwen3.8-27B": ("bfloat16", 8, [2048, 4096]),
            "Qwen/Qwen3.5-9B": ("float32", 4, [4096])}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def hashed_files(root, names):
    files = {}
    for name in sorted(names):
        if Path(name).name != name:
            raise ValueError(f"expected a direct checkpoint filename: {name}")
        path = root / name
        before = path.stat()
        print(f"Hashing {name} ({before.st_size} bytes)", flush=True)
        checksum = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"source changed while hashing: {name}")
        files[name] = {"bytes": after.st_size, "sha256": checksum}
    return files


def model_weights(root, kind):
    if kind == "packed":
        return hashed_files(root, ("artifact.json", "weights.bin"))
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        names.add(index.name)
    else:
        names = {"model.safetensors"}
    return hashed_files(root, names)


def validate_report(report, common, problems, scorer, summarize, kind):
    fingerprint = report["fingerprint"]
    if (report.get("schema") != 1 or report.get("kind") != kind
            or fingerprint["common"] != common
            or report.get("fingerprint_sha256") != digest(fingerprint)
            or report.get("state") not in ("running", "completed")):
        raise ValueError(f"{kind} report fingerprint/schema differs; refusing to overwrite")
    caps = report["caps"]
    if set(caps) != {str(cap) for cap in common["max_new_tokens"]}:
        raise ValueError(f"{kind} cap set changed")
    by_id = {p["id"]: p for p in problems}
    ordered = [key for batch in common["batch_ids"] for key in batch]
    boundaries = [0]
    for batch in common["batch_ids"]:
        boundaries.append(boundaries[-1] + len(batch))
    unfinished = False
    for cap in common["max_new_tokens"]:
        saved = caps[str(cap)]
        rows = saved["rows"]
        n = len(rows)
        if (n not in boundaries or set(rows) != set(ordered[:n])
                or saved["completed_batches"] != boundaries.index(n)
                or (unfinished and n)):
            raise ValueError(f"{kind}/{cap}: rows must be a prefix of complete original batches")
        if (saved["state"] not in ("pending", "running", "completed")
                or (saved["state"] == "completed") != (n == len(problems))
                or (saved["state"] == "pending" and n)
                or not isinstance(saved["seconds"], (int, float))
                or not math.isfinite(saved["seconds"]) or saved["seconds"] < 0):
            raise ValueError(f"{kind}/{cap}: invalid state or timing")
        for key, row in rows.items():
            tokens = row["tokens"]
            if (type(tokens) is not int or not 0 <= tokens <= cap
                    or type(row["correct"]) is not bool or type(row["finished"]) is not bool
                    or row["finished"] != (tokens < cap) or not isinstance(row["text"], str)
                    or row["end"] != row["text"][-200:]
                    or row["question"] != by_id[key]["question"]
                    or row["expected_answer"] != by_id[key]["answer"]):
                raise ValueError(f"{kind}/{cap}: invalid saved row {key}")
            correct, answer = scorer(by_id[key], row["text"]) if tokens < cap else (False, None)
            if (row["correct"], row["answer"]) != (correct, answer):
                raise ValueError(f"{kind}/{cap}: saved score differs for {key}")
        expected = summarize([by_id[key] for key in ordered[:n]], rows) if n else None
        if saved["summary"] != expected:
            raise ValueError(f"{kind}/{cap}: summary differs from saved rows")
        unfinished |= n < len(problems)
    if (report["state"] == "completed") != all(c["state"] == "completed" for c in caps.values()):
        raise ValueError(f"{kind}: inconsistent completion state")


def comparison(output, reports, common):
    result = {"schema": 1, "state": "running", "common": common,
              "common_sha256": digest(common), "reports": {}, "caps": {}}
    for kind, report in reports.items():
        result["reports"][kind] = {"file": f"{kind}.json",
                                   "sha256": file_sha256(output / f"{kind}.json"),
                                   "fingerprint_sha256": report["fingerprint_sha256"]}
    for cap in common["max_new_tokens"]:
        item = {}
        for kind in ("original", "packed"):
            saved = reports.get(kind, {}).get("caps", {}).get(str(cap))
            item[kind + "_completed_rows"] = len(saved["rows"]) if saved else 0
            item[kind] = saved["summary"] if saved and saved["state"] == "completed" else None
        ready = item["original"] is not None and item["packed"] is not None
        original_right = item["original"]["right"] if ready else 0
        item.update(state="completed" if ready else "running",
                    relative_retention=item["packed"]["right"] / original_right
                    if ready and original_right else None)
        if ready and not original_right:
            item["retention_note"] = "Original score is zero; relative retention is undefined."
        result["caps"][str(cap)] = item
    result["state"] = ("completed" if all(c["state"] == "completed" for c in result["caps"].values())
                       else "running")
    return result


def sync_reports(output, remote):
    if not remote:
        return
    # Only these exact report names are eligible; logs, models and credentials
    # cannot be copied even if a caller places other files in the output folder.
    filters = [part for name in REPORTS for part in ("--filter", "+ /" + name)]
    filters += ["--filter", "- **"]
    log = output.parent / (output.name + ".rclone.log")
    with log.open("ab") as stream:
        for command in (
            ["rclone", "copy", str(output), remote, *filters, "--transfers", "1", "--checkers", "2"],
            ["rclone", "check", str(output), remote, *filters, "--one-way", "--checkers", "2"],
        ):
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("Small-report rclone copy/check completed", flush=True)


def persist(output, report, reports, common, remote):
    report["state"] = ("completed" if all(c["state"] == "completed" for c in report["caps"].values())
                       else "running")
    _write_json(output / f"{report['kind']}.json", report)
    reports[report["kind"]] = report
    result = comparison(output, reports, common)
    _write_json(output / "comparison.json", result)
    sync_reports(output, remote)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=["original", "packed"], required=True)
    parser.add_argument("--model-path", type=Path, help="original local HF snapshot; required for original")
    parser.add_argument("--packed", type=Path, help="artifact.json/weights.bin directory; required for packed")
    parser.add_argument("--model-id", choices=list(SCHEDULE), required=True, help="base Qwen model ID")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--max-new", type=int, choices=[2048, 4096], nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="dedicated report directory; identical args resume")
    parser.add_argument("--rclone-remote", help="dedicated remote:subdirectory for the small JSON reports only")
    args = parser.parse_args()
    if len(args.max_new) != len(set(args.max_new)):
        parser.error("--max-new contains duplicates")
    args.max_new = sorted(args.max_new)
    if (args.dtype, args.batch, args.max_new) != SCHEDULE[args.model_id]:
        parser.error("this run uses 27B BF16/batch8/caps2048+4096 or 9B FP32/batch4/cap4096")
    source = args.model_path if args.kind == "original" else args.packed
    if source is None:
        parser.error("original needs --model-path; packed needs --packed")
    source = source.resolve()
    if not (source / "config.json").is_file():
        parser.error("the selected model directory needs config.json")
    if args.rclone_remote:
        remote, separator, subdirectory = args.rclone_remote.partition(":")
        if not remote or not separator or not subdirectory.strip("/") or shutil.which("rclone") is None:
            parser.error("--rclone-remote needs an installed rclone and a dedicated remote:subdirectory")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    # Lock and rclone logs live outside the report folder and are never synced.
    import fcntl
    with (args.output.parent / (args.output.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import numpy
        import torch
        import transformers
        from transformers import AutoTokenizer
        from math_eval import B, solve, summary
        from load_packed import load_packed

        if not torch.cuda.is_available():
            parser.error("this matched evaluation requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        problems = json.loads(PROTOCOL.read_text())["problems"]
        if len(problems) != 216 or len({p["id"] for p in problems}) != 216 or not all(B.check(p) for p in problems):
            raise ValueError("expected the unchanged 216-problem protocol")
        tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Preserve solve's global ordering before calling it one complete batch
        # at a time, so incremental checkpoints do not change batch membership.
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{p['question']}\n\n{B.INSTRUCTION}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in problems]
        order = sorted(range(len(problems)), key=lambda i: (len(prompts[i]), problems[i]["id"]))
        ordered = [problems[i] for i in order]
        batches = [ordered[start:start + args.batch] for start in range(0, len(ordered), args.batch)]
        common = {"model_id": args.model_id, "dtype": args.dtype, "batch": args.batch,
                  "max_new_tokens": args.max_new, "thinking": False, "greedy": True, "tf32": False, "seed": 0,
                  "protocol_sha256": file_sha256(PROTOCOL),
                  "text_assets": hashed_files(source, COMMON_TEXT_ASSETS),
                  "generation_config": (hashed_files(source, ("generation_config.json",))["generation_config.json"]
                                        if (source / "generation_config.json").is_file() else None),
                  "source_sha256": {name: file_sha256(SOURCE / name) for name in SOURCES},
                  "batch_ids": [[p["id"] for p in batch] for batch in batches],
                  "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                                  "numpy": numpy.__version__, "cuda": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(0)}}
        fingerprint = {
            "common": common, "weights": model_weights(source, args.kind),
            "optional_text_assets": hashed_files(
                source, [name for name in OPTIONAL_TEXT_ASSETS if (source / name).is_file()]),
        }
        reports = {}
        for kind in ("original", "packed"):
            path = args.output / f"{kind}.json"
            if path.exists():
                reports[kind] = json.loads(path.read_text())
                validate_report(reports[kind], common, problems, B.score, summary, kind)
        old_comparison = args.output / "comparison.json"
        if old_comparison.exists():
            old = json.loads(old_comparison.read_text())
            if old.get("common") != common or any(kind not in reports for kind in old.get("reports", {})):
                raise ValueError("comparison fingerprint/source report differs; refusing to overwrite")
        report = reports.get(args.kind)
        if report is not None and report["fingerprint"] != fingerprint:
            raise ValueError("model weights/config/source fingerprint changed; refusing to overwrite")
        if report is None:
            report = {"schema": 1, "kind": args.kind, "state": "running",
                      "fingerprint": fingerprint, "fingerprint_sha256": digest(fingerprint),
                      "scope": "Math-only; each cap is generated independently; no PPL/parity/vision rerun",
                      "caps": {str(cap): {"state": "pending", "completed_batches": 0, "seconds": 0.0,
                                          "summary": None, "rows": {}} for cap in args.max_new}}
        persist(args.output, report, reports, common, args.rclone_remote)
        if report["state"] != "completed":
            model = (load_original_model(source, dtype=args.dtype, device="cuda")
                     if args.kind == "original" else
                     load_packed(source, dtype=args.dtype, device="cuda", decode_device="cuda"))
            try:
                by_id = {p["id"]: p for p in problems}
                for cap in args.max_new:
                    saved = report["caps"][str(cap)]
                    for batch in batches[saved["completed_batches"]:]:
                        started = time.perf_counter()
                        rows = solve(model, tokenizer, batch, args.batch, cap, thinking=False, keep_text=True)
                        if set(rows) != {p["id"] for p in batch}:
                            raise ValueError("solve did not return one complete requested batch")
                        for key, row in rows.items():
                            row.update(question=by_id[key]["question"], expected_answer=by_id[key]["answer"])
                        saved["rows"].update(rows)
                        saved["completed_batches"] += 1
                        saved["seconds"] += time.perf_counter() - started
                        done = len(saved["rows"])
                        saved["state"] = "completed" if done == len(problems) else "running"
                        saved["summary"] = summary([p for p in ordered if p["id"] in saved["rows"]], saved["rows"])
                        persist(args.output, report, reports, common, args.rclone_remote)
                        print(f"{args.kind} cap={cap}: {done}/216; "
                              f"{saved['summary']['right']} right, "
                              f"{saved['summary']['hit the token limit']} at cap", flush=True)
            finally:
                del model
                torch.cuda.empty_cache()
        result = comparison(args.output, reports, common)
        for cap, item in result["caps"].items():
            counts = {kind: ({"right": item[kind]["right"], "of": item[kind]["of"],
                             "hit_token_limit": item[kind]["hit the token limit"]}
                            if item[kind] else None) for kind in ("original", "packed")}
            print(f"cap={cap}: {counts}; relative_retention={item['relative_retention']}", flush=True)


if __name__ == "__main__":
    main()
