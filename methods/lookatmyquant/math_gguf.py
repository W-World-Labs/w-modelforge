"""Independent native llama.cpp math216, using the existing protocol and scorer.

Original BF16 GGUF and adaptive GGUF run separately with identical arguments.
HF-rendered prompt token IDs bypass chat-template differences. Complete cohorts
are saved atomically; native continuous batching and F16 KV are explicit.
"""
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
# Reuses Shin Megami Boson's math_bench exact-answer scorer (MIT).
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

import math_bench as B
from compare_math_caps import digest
from row_allocation import _write_json, file_sha256

SOURCE = Path(__file__).resolve().parent


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def totals(rows):
    return {"right": sum(r["correct"] for r in rows.values()), "of": len(rows),
            "hit_token_limit": sum(not r["finished"] for r in rows.values()),
            "mean_tokens": sum(r["tokens"] for r in rows.values()) / len(rows) if rows else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("original", "adaptive"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="directory containing the published red/blue examples")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=4096)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--context-per-slot", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if args.max_new != 4096 or args.parallel != 4 or args.context_per_slot != 8192:
        parser.error("current native profile is cap4096/parallel4/context8192 per slot")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = args.out.with_suffix(".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from transformers import AutoTokenizer, __version__ as transformers_version
    from gguf import GGUFReader
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    protocol = SOURCE / "results/protocols/math216.json"
    problems = json.loads(protocol.read_text())["problems"]
    by_id = {p["id"]: p for p in problems}
    if len(problems) != 216 or len(by_id) != 216:
        raise ValueError("expected frozen216 protocol")
    prompts = {p["id"]: tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{p['question']}\n\n{B.INSTRUCTION}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in problems}
    ordered = sorted(by_id, key=lambda key: (len(prompts[key]), key))
    prompt_ids = {key: tokenizer.encode(prompts[key], add_special_tokens=False) for key in ordered}
    if max(map(len, prompt_ids.values())) + args.max_new >= args.context_per_slot:
        raise ValueError("prompt plus generation exceeds a native slot")
    binary = args.llama_dir / "build/bin/llama-server"
    engine_commit = subprocess.check_output(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"], text=True).strip()
    reader = GGUFReader(str(args.gguf))
    stored_types = dict(Counter(t.tensor_type.name for t in reader.tensors))
    del reader
    fingerprint = {"model_id": args.model_id, "kind": args.kind,
        "protocol_sha256": file_sha256(protocol), "gguf_sha256": file_sha256(args.gguf),
        "mmproj_sha256": file_sha256(args.mmproj), "engine_commit": engine_commit,
        "server_sha256": file_sha256(binary), "driver_sha256": file_sha256(Path(__file__)),
        "gguf_tensor_types": stored_types,
        "gpu_and_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
        "scorer_sha256": file_sha256(SOURCE / "math_bench.py"), "prompt_ids_sha256": digest(prompt_ids),
        "text_assets": {name: file_sha256(args.snapshot / name) for name in
                        ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "transformers": transformers_version, "max_new_tokens": args.max_new,
        "parallel": args.parallel, "context_per_slot": args.context_per_slot, "kv_types": "F16/F16",
        "seed": 0, "samplers": ["temperature"], "temperature": 0, "cache_prompt": False,
        "thinking": False, "ordered_ids": ordered,
        "images": {name: file_sha256(args.images / name) for name in
                   ("vision-red-square.png", "vision-blue-circle.png")}}
    report = {"schema": 1, "state": "running", "fingerprint": fingerprint,
              "fingerprint_sha256": digest(fingerprint), "rows": {}, "image_checks": [],
              "scope": "Native llama.cpp quality, fresh original BF16/adaptive GGUF. Different backend from Torch LAMQ reports; no scores or speed claims inherited."}
    if args.out.exists():
        report = json.loads(args.out.read_text())
        if report["fingerprint"] != fingerprint or report["fingerprint_sha256"] != digest(fingerprint):
            raise ValueError("resume fingerprint differs")
        rows = report["rows"]
        if len(rows) % args.parallel or set(rows) != set(ordered[:len(rows)]):
            raise ValueError("resume is not a complete-cohort prefix")
        for key, row in rows.items():
            raw = row["generated_token_ids"]
            eos = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
            n = min(next((i for i, token in enumerate(raw) if token in eos), len(raw)), args.max_new)
            if (row["question"] != by_id[key]["question"] or row["expected_answer"] != by_id[key]["answer"]
                    or row["tokens"] != n or row["text"] != tokenizer.decode(raw[:n], skip_special_tokens=False)
                    or row["stop_type"] not in ("eos", "limit")
                    or row["finished"] != (n < args.max_new and row["stop_type"] == "eos")):
                raise ValueError("saved native row differs from recorded generation")
            score = B.score(by_id[key], row["text"]) if row["finished"] else (False, None)
            if (row["correct"], row["answer"]) != score:
                raise ValueError("saved score differs")
        if report["state"] == "completed":
            print(json.dumps(report["summary"])); return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = [str(binary), "-m", str(args.gguf), "--mmproj", str(args.mmproj),
        "--host", "127.0.0.1", "--port", str(args.port), "--alias", "w-modelforge",
        "--ctx-size", str(args.context_per_slot * args.parallel), "--parallel", str(args.parallel),
        "--n-gpu-layers", "99", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--jinja", "--reasoning", "off", "--no-warmup", "--no-context-shift", "--seed", "0"]
    report["command"] = command
    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("HF_TOKEN", None)
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    process = None
    started = time.monotonic()
    try:
        with args.out.with_suffix(".server.log").open("ab") as log:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        while True:
            if process.poll() is not None:
                raise RuntimeError("native server exited during startup")
            try:
                request(args.port, "/health"); break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("native server health timeout")
                time.sleep(1)
        # Verify the GGUF tokenizer's special-token parsing against all HF prompts.
        for key in ordered:
            got = request(args.port, "/tokenize", {"content": prompts[key], "add_special": False, "parse_special": True})["tokens"]
            if got != prompt_ids[key]:
                raise ValueError(f"native/HF prompt token IDs differ: {key}")
        report["all216_native_tokenizations_match"] = True
        if not report["image_checks"]:
            for name, expected in (("vision-red-square.png", "red"), ("vision-blue-circle.png", "blue")):
                image = base64.b64encode((args.images / name).read_bytes()).decode()
                result = request(args.port, "/v1/chat/completions", {"model": "w-modelforge",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Name the color of the large shape. Answer with one lowercase color word."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]}],
                    "temperature": 0, "seed": 0, "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False}})
                answer = result["choices"][0]["message"]["content"]
                passed = re.fullmatch(expected + r"[.!]?", answer.strip().lower()) is not None
                report["image_checks"].append({"image": name, "expected": expected, "answer": answer, "passed": passed})
            _write_json(args.out, report)
        if not all(row["passed"] for row in report["image_checks"]):
            raise RuntimeError("basic native image response did not match")

        def solve(slot_key):
            slot, key = slot_key
            result = request(args.port, "/completion", {"prompt": prompt_ids[key],
                "n_predict": args.max_new, "return_tokens": True, "stream": False,
                "samplers": ["temperature"], "temperature": 0, "seed": 0,
                "cache_prompt": False, "id_slot": slot})
            if result["stop_type"] not in ("eos", "limit") or result.get("truncated"):
                raise RuntimeError(f"unexpected native stopping/context behavior: {key}")
            raw = result["tokens"]
            if not isinstance(raw, list) or any(type(token) is not int for token in raw):
                raise RuntimeError("native response has no generated token IDs")
            eos = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
            n = next((i for i, token in enumerate(raw) if token in eos), len(raw))
            n = min(n, args.max_new)
            text = tokenizer.decode(raw[:n], skip_special_tokens=False)
            finished = n < args.max_new and result["stop_type"] == "eos"
            correct, answer = B.score(by_id[key], text) if finished else (False, None)
            return key, {"question": by_id[key]["question"], "expected_answer": by_id[key]["answer"],
                "correct": correct, "answer": answer, "tokens": n, "finished": finished,
                "text": text, "generated_token_ids": raw, "api_content": result["content"],
                "stop_type": result["stop_type"], "native_timings": result.get("timings")}

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for start in range(len(report["rows"]), len(ordered), args.parallel):
                batch = ordered[start:start + args.parallel]
                report["rows"].update(pool.map(solve, enumerate(batch)))
                report["summary"] = totals(report["rows"])
                report["elapsed_seconds_this_process"] = time.monotonic() - started
                _write_json(args.out, report)
                print(json.dumps(report["summary"]), flush=True)
        report["state"] = "completed"
        _write_json(args.out, report)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


if __name__ == "__main__":
    main()
