"""Expand a released packed model into a standard, full multimodal HF checkpoint.

This preserves the language build dtype and original BF16 vision. It does not
produce native low-bit vLLM weights or a GGUF file.
"""

# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path, help="new directory; never overwrites an existing checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--decode-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    evaluation = json.loads((args.release / "evaluation.json").read_text())
    vision = json.loads((args.release / "vision-provenance.json").read_text())
    if (vision["quantized"] or vision["dtype_conversion"]
            or vision["source_model"] != evaluation["model"]
            or vision["source_revision"] != evaluation["base_revision"]):
        parser.error("original vision provenance does not match this release")
    vision_path = args.release / "vision.safetensors"
    digest = hashlib.sha256()
    with vision_path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != vision["weights_sha256"]:
        parser.error("original vision SHA256 mismatch")

    from transformers import AutoProcessor
    from load_packed import load_packed

    model = load_packed(args.release / "packed", dtype=evaluation["build_dtype"],
                        device=args.device, decode_device=args.decode_device, vision_file=vision_path)
    # Serialize on CPU, retaining per-parameter dtypes and the original visual values.
    model.to("cpu")
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size=args.max_shard_size)
    # save_pretrained infers the top-level dtype from the first parameter, which
    # can be BF16 vision even when the language checkpoint is FP32.
    model.config.dtype = evaluation["build_dtype"]
    model.config.text_config.dtype = evaluation["build_dtype"]
    model.config.vision_config.dtype = "bfloat16"
    model.config.save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.release, local_files_only=True).save_pretrained(args.output)
    for name in ("LICENSE", "ATTRIBUTION.md", "MODEL_MODIFICATIONS.md"):
        source = args.release / name
        if source.is_file():
            shutil.copyfile(source, args.output / name)
    receipt = {"source_model": evaluation["model"], "source_revision": evaluation["base_revision"],
               "language_dtype": evaluation["build_dtype"], "visual_dtype": "bfloat16",
               "vision_source_sha256": vision["weights_sha256"],
               "language_manifest_sha256": hashlib.sha256((args.release / "packed/artifact.json").read_bytes()).hexdigest(),
               "format": "standard dense Hugging Face safetensors; expanded quantized values",
               "warning": "Engine dtype conversion, GGUF conversion and subsequent requantization need separate evaluation."}
    (args.output / "w-modelforge-export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Expanded full multimodal checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
