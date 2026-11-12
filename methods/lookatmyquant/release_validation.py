"""Evaluate actual packed weights against a dense build before release.

Run reference and packed in separate processes. This writes evaluation reports,
not a publication approval, completion marker, or Hugging Face upload.
"""

# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
# Reuses Shin Megami Boson's LookAtMyQuant model evaluation and packed decoding (MIT).

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from compiler_artifact import file_sha256
from load_packed import load_packed
from math_eval import B, load_dense, solve, summary, token_ids
from row_allocation import _write_json, load_original_model

PROTOCOL = Path(__file__).parent / "results/protocols/math216.json"
PROMPTS = [
    ("english", "Explain in two sentences why the sky appears blue."),
    ("chinese", "请用两句话说明量化模型为什么能节省内存。"),
    ("code", "Write a Python function that returns the sum of the squares of a list of integers."),
    ("math", "Compute 123 * 456 and briefly show your calculation."),
]
SCOPE = "Additional 4x128 WikiText-2 validation windows; not the main 8/16x512 evaluation."


@torch.no_grad()
def _capture(model, ids):
    probabilities, loss = [], 0.0
    for index, window in enumerate(ids):
        x = window[None].to("cuda")
        lp = F.log_softmax(model(x, use_cache=False).logits.float(), -1)
        if not bool(torch.isfinite(lp).all()):
            raise ValueError("nonfinite model log probabilities")
        loss -= float(lp[:, :-1].gather(-1, x[:, 1:, None]).sum())
        probabilities.append(lp.cpu())
        print(f"Captured window {index + 1}/{len(ids)}", flush=True)
    return torch.cat(probabilities), math.exp(loss / (ids.shape[0] * (ids.shape[1] - 1)))


def _compare(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError("reference and actual distributions have different shapes")
    total = reference.shape[0] * reference.shape[1]
    kl_sum = kl_max = absolute = squared = maximum = same = 0.0
    for ref_window, got_window in zip(reference, actual):
        ref, got = ref_window.double(), got_window.double()
        ref = ref - torch.logsumexp(ref, -1, keepdim=True)
        got = got - torch.logsumexp(got, -1, keepdim=True)
        error = got - ref
        kl = (ref.exp() * (ref - got)).sum(-1)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
        absolute += float(error.abs().sum())
        squared += float(error.square().sum())
        maximum = max(maximum, float(error.abs().max()))
        same += int((ref.argmax(-1) == got.argmax(-1)).sum())
    entries = reference.numel()
    return {"kl_reference_to_actual_mean": kl_sum / total, "kl_reference_to_actual_max": kl_max,
            "logprob_abs_mean": absolute / entries, "logprob_abs_max": maximum,
            "logprob_rmse": math.sqrt(squared / entries), "top1_agreement": same / total}


@torch.no_grad()
def _generate(model, tokenizer):
    results = []
    for kind, prompt in PROMPTS:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        generated = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   max_new_tokens=32, repetition_penalty=1.0, use_cache=True,
                                   pad_token_id=(tokenizer.eos_token_id if tokenizer.pad_token_id is None
                                                 else tokenizer.pad_token_id))
        new = generated[0, inputs.input_ids.shape[1]:].cpu()
        results.append({"kind": kind, "prompt": prompt, "input_ids": inputs.input_ids[0].cpu().tolist(),
                        "new_tokens": new.tolist(), "text": tokenizer.decode(new, skip_special_tokens=True)})
        print(f"Generated {kind} prompt ({len(new)} tokens)", flush=True)
    return results


def reference_phase(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    ids = token_ids(tokenizer, "validation", args.cache_tag).cpu()
    windows = ids[:len(ids) // 128 * 128].reshape(-1, 128)
    if len(windows) < 4:
        raise ValueError("WikiText validation cache has fewer than four windows")
    ids = windows[torch.randperm(len(windows), generator=torch.Generator().manual_seed(0))[:4]].clone()
    model = load_original_model(args.model_path, dtype=args.dtype, device="cuda")
    original, original_ppl = _capture(model, ids)
    load = load_dense(model, args.dense)
    dense, dense_ppl = _capture(model, ids)
    prompts = _generate(model, tokenizer)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    payload = {"ids": ids, "original_logprobs": original, "dense_logprobs": dense, "dense_prompts": prompts}
    target = args.work_dir / "release-reference.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    print("Hashing reference cache and dense checkpoint after releasing the GPU model", flush=True)
    receipt = {"phase": "reference", "dtype": args.dtype, "cache_tag": args.cache_tag, "scope": SCOPE,
               "windows": list(ids.shape), "window_seed": 0, "max_prompt_new_tokens": 32,
               "ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
               "reference_file": target.name, "reference_sha256": file_sha256(target),
               "model_snapshot": Path(args.model_path).name,
               "config_sha256": file_sha256(Path(args.model_path) / "config.json"),
               "dense_sha256": file_sha256(args.dense), "dense_load": load,
               "math_protocol_sha256": file_sha256(PROTOCOL),
               "original_ppl": original_ppl, "dense_ppl": dense_ppl,
               "original_to_dense": _compare(original, dense)}
    _write_json(args.work_dir / "release-reference.json", receipt)
    return receipt


def packed_phase(args):
    receipt = json.loads((args.work_dir / "release-reference.json").read_text())
    reference_file = args.work_dir / "release-reference.pt"
    if (receipt["dtype"] != args.dtype or receipt["cache_tag"] != args.cache_tag
            or receipt["reference_sha256"] != file_sha256(reference_file)
            or receipt["math_protocol_sha256"] != file_sha256(PROTOCOL)
            or receipt["config_sha256"] != file_sha256(args.packed / "config.json")):
        raise ValueError("reference dtype, tokenizer namespace, protocol, config or file hash changed")
    reference = torch.load(reference_file, map_location="cpu", mmap=True, weights_only=True)
    ids = reference["ids"]
    if (list(ids.shape) != [4, 128]
            or hashlib.sha256(ids.numpy().tobytes()).hexdigest() != receipt["ids_sha256"]):
        raise ValueError("reference window identity changed")
    problems = json.loads(PROTOCOL.read_text())["problems"]
    if len(problems) != 216 or not all(B.check(problem) for problem in problems):
        raise ValueError("expected the unchanged independent 216-question math protocol")
    baseline = baseline_hash = None
    if args.baseline_math:
        raw = args.baseline_math.read_bytes()
        baseline, baseline_hash = json.loads(raw), hashlib.sha256(raw).hexdigest()
        original_rows = baseline["models"]["orig"]["rows"]
        if (set(original_rows) != {problem["id"] for problem in problems}
                or baseline["max new tokens"] != args.max_new or baseline["thinking"]
                or not baseline["greedy"] or baseline.get("dtype", "float32") != args.dtype):
            raise ValueError("baseline math must use the same ids, dtype, token limit and greedy no-thinking recipe")
    model = load_packed(args.packed, dtype=args.dtype, device="cuda", decode_device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.packed, padding_side="left", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    actual, ppl = _capture(model, ids)
    prompts = _generate(model, tokenizer)
    prompt_checks = []
    for expected, actual_prompt in zip(reference["dense_prompts"], prompts):
        if expected["kind"] != actual_prompt["kind"] or expected["input_ids"] != actual_prompt["input_ids"]:
            raise ValueError("packed tokenizer changed a fixed prompt")
        prompt_checks.append({"kind": expected["kind"], "prompt": expected["prompt"],
                              "same_greedy_tokens": expected["new_tokens"] == actual_prompt["new_tokens"],
                              "dense_text": expected["text"], "packed_text": actual_prompt["text"]})
    report = {"phase": "packed", "dtype": args.dtype, "scope": SCOPE, "windows": list(ids.shape),
              "ids_sha256": receipt["ids_sha256"], "reference_sha256": receipt["reference_sha256"],
              "dense_sha256": receipt["dense_sha256"], "model_snapshot": receipt["model_snapshot"],
              "packed_manifest_sha256": file_sha256(args.packed / "artifact.json"),
              "packed_weights_sha256": file_sha256(args.packed / "weights.bin"),
              "math_protocol_sha256": receipt["math_protocol_sha256"],
              "original_ppl": receipt["original_ppl"], "dense_ppl": receipt["dense_ppl"], "packed_ppl": ppl,
              "packed_to_original_ppl_ratio": ppl / receipt["original_ppl"],
              "dense_to_packed": _compare(reference["dense_logprobs"], actual),
              "original_to_dense": receipt["original_to_dense"],
              "original_to_packed": _compare(reference["original_logprobs"], actual),
              "prompts": prompt_checks, "max_prompt_new_tokens": 32,
              "math": {"state": "running", "problems": len(problems), "batch": args.batch,
                       "max_new_tokens": args.max_new, "thinking": False, "greedy": True},
              "publication": "Not decided by this CLI; math retention and PPL/KL require release review."}
    _write_json(args.out, report)
    rows = solve(model, tokenizer, problems, args.batch, args.max_new, thinking=False)
    report["math"].update(state="completed", summary=summary(problems, rows), rows=rows)
    if baseline is not None:
        original_right = sum(bool(row["correct"]) for row in baseline["models"]["orig"]["rows"].values())
        if not original_right:
            raise ValueError("original math score is zero; retention is undefined")
        retention = report["math"]["summary"]["right"] / original_right
        report["math"].update(original_right=original_right, retention=retention,
                               minimum_retention=0.9, retention_passed=retention >= 0.9,
                               baseline_sha256=baseline_hash)
    else:
        report["math"].update(retention=None, retention_passed=None, reason="--baseline-math was not supplied")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["reference", "packed"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dense", type=Path)
    parser.add_argument("--packed", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--cache-tag", default="qwen35")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-new", type=int, choices=[1024, 2048], default=1024)
    parser.add_argument("--baseline-math", type=Path)
    args = parser.parse_args()
    if ((args.phase == "reference" and (args.model_path is None or args.dense is None))
            or (args.phase == "packed" and args.packed is None)):
        parser.error("reference needs --model-path/--dense; packed needs --packed")
    if args.batch < 1:
        parser.error("--batch must be positive")
    if not torch.cuda.is_available():
        parser.error("this release evaluation uses CUDA, like the main Colab math evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = reference_phase(args) if args.phase == "reference" else packed_phase(args)
    report.update(seconds=round(time.time() - started, 1), torch_version=torch.__version__, tf32=False)
    _write_json(args.out, report)
    print(f"Saved {args.phase} evaluation: {args.out}", flush=True)


if __name__ == "__main__":
    main()
