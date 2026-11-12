"""Load opgraph-packed-1 weights into a dense Transformers text model.

This reference loader needs the packed directory and its config/tokenizer assets,
not the original checkpoint. It expands weights; it is not a packed inference kernel.
"""

# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
# Packed decoding and Hadamard rules: Shin Megami Boson, MIT.

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from compiler_artifact import decode_rotated, read_artifact
from opgraph.hadamard import hadamard_factors


def _tensor(tensors, manifest, name):
    value = torch.from_numpy(np.array(tensors[name], copy=True))
    if manifest["tensors"][name]["dtype"] == "BF16":
        value = value.view(torch.bfloat16)
    return value


def _decode_matrix(tensors, manifest, matrix, dtype, decode_device, chunk_rows):
    shape = tuple(matrix["shape"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("packed matrix must have a nonempty two-dimensional shape")
    result = torch.empty(shape, dtype=dtype, device="cpu")
    if matrix["kind"] == "table":
        row_ids = []
        for bucket in matrix["buckets"]:
            rows = torch.from_numpy(np.array(tensors[bucket["rows"]], dtype=np.int64, copy=True))
            row_ids.append(rows)
            value = (_decode_matrix(tensors, manifest, bucket["matrix"], dtype, decode_device, chunk_rows)
                     if "matrix" in bucket else _tensor(tensors, manifest, bucket["dense"]))
            if rows.ndim != 1 or value.shape != (len(rows), shape[1]):
                raise ValueError("packed table bucket shape mismatch")
            result[rows] = value.to(dtype)
        if not row_ids or not torch.equal(torch.cat(row_ids).sort().values, torch.arange(shape[0])):
            raise ValueError("packed table buckets must cover every row exactly once")
        return result
    if matrix["kind"] != "grouped":
        raise ValueError(f"unsupported packed matrix kind: {matrix['kind']}")
    groups = matrix["groups"]
    row_ids = [np.asarray(tensors[group["key"] + ".rows"], dtype=np.int64) for group in groups]
    if (not row_ids or any(rows.ndim != 1 for rows in row_ids)
            or not np.array_equal(np.sort(np.concatenate(row_ids)), np.arange(shape[0]))
            or any(group["columns"] != shape[1] for group in groups)):
        raise ValueError("packed groups must cover every matrix row at the recorded width")
    signs = torch.from_numpy(np.array(tensors[matrix["signs"]], copy=True)).to(decode_device, torch.float64)
    if signs.shape != (shape[1],) or not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError("invalid packed input rotation signs")
    rotated = decode_rotated(tensors, matrix)
    A, B = (factor.to(decode_device) for factor in hadamard_factors(shape[1]))
    # D @ (A ⊗ B) = vec(A.T @ reshape(D) @ B), then the recorded input signs.
    # Reuse the quantizer's exact factors without allocating a full n-by-n rotation.
    for start in range(0, shape[0], chunk_rows):
        stop = min(start + chunk_rows, shape[0])
        block = torch.from_numpy(rotated[start:stop]).to(decode_device, torch.float64)
        block = (A.T @ block.reshape(-1, len(A), len(B))) @ B
        block = block.reshape(stop - start, shape[1]) / math.sqrt(shape[1]) * signs
        result[start:stop].copy_(block.to(device="cpu", dtype=dtype))
    return result


@torch.no_grad()
def load_packed(path, *, dtype="float32", device="cpu", decode_device="cpu", chunk_rows=256):
    """Restore all text model weights from a packed directory, with no base weights.

    dtype must match the build for a dense.pt parity comparison. The model is
    normally initialized on CPU to preserve nonpersistent rotary buffers. Only
    the current matrix's decoding uses decode_device; device is the final model
    destination. All checksums, state coverage, shapes and finite values are checked.
    """
    path = Path(path)
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    if dtype not in (torch.float32, torch.bfloat16) or chunk_rows < 1:
        raise ValueError("use float32/bfloat16 and positive chunk_rows")
    decode_device = torch.device(decode_device)
    if decode_device.type not in ("cpu", "cuda"):
        raise ValueError("FP64 decoding requires a CPU or CUDA device")
    tensors, manifest = read_artifact(path)
    if not manifest.get("self_contained_weights"):
        raise ValueError("artifact does not declare a complete model state")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
    if (path / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    model.requires_grad_(False)
    state = model.state_dict()
    matrices, dense, aliases = manifest["matrices"], set(manifest["dense"]), manifest.get("aliases", {})
    if (set(matrices) & dense or (set(matrices) | dense) & set(aliases)
            or set(state) != set(matrices) | dense | set(aliases)):
        missing = sorted(set(state) - (set(matrices) | dense | set(aliases)))
        extra = sorted((set(matrices) | dense | set(aliases)) - set(state))
        raise ValueError(f"packed state coverage mismatch; missing={missing}, extra={extra}")
    for name, target in state.items():
        if name in aliases:
            continue
        value = (_decode_matrix(tensors, manifest, matrices[name], target.dtype, decode_device, chunk_rows)
                 if name in matrices else _tensor(tensors, manifest, name))
        if value.shape != target.shape:
            raise ValueError(f"packed state shape mismatch: {name}")
        value = value.to(target.dtype)
        flat = value.reshape(-1)
        if any(not bool(torch.isfinite(flat[start:start + (1 << 22)]).all())
               for start in range(0, flat.numel(), 1 << 22)):
            raise ValueError(f"nonfinite restored weight: {name}")
        target.copy_(value)
        print(f"Restored {name} {tuple(target.shape)}", flush=True)
        del value, flat
    for name, owner in aliases.items():
        if owner not in matrices and owner not in dense:
            raise ValueError(f"packed alias needs a non-alias owner: {name}")
        if state[name].shape != state[owner].shape or state[name].dtype != state[owner].dtype:
            raise ValueError(f"packed alias shape/dtype mismatch: {name}")
        module_name, leaf = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        try:
            parameter = model.get_parameter(owner)
        except AttributeError:
            module.register_buffer(leaf, model.get_buffer(owner))
        else:
            module.register_parameter(leaf, parameter)
    del state, tensors
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="directory containing artifact.json, weights.bin and model assets")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode-device", default="cpu")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--prompt", action="append", required=True, help="repeat for multiple prompts")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    model = load_packed(args.path, dtype=args.dtype, device=args.device,
                        decode_device=args.decode_device, chunk_rows=args.chunk_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.path, local_files_only=True)
    for prompt in args.prompt:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                    pad_token_id=(tokenizer.eos_token_id if tokenizer.pad_token_id is None
                                                  else tokenizer.pad_token_id))
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
