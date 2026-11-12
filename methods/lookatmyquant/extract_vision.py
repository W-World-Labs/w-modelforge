"""Preserve original Qwen3.5/3.8 vision tensors and processor assets without quantization."""

# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT

from pathlib import Path
import argparse
import hashlib
import json
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compiler_artifact import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.model_path, args.out
    config = json.loads((source / "config.json").read_text())
    if not config.get("vision_config"):
        raise ValueError("Expected the original multimodal config with vision_config")
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(name for name in weight_map if name.startswith("model.visual."))
    if not names:
        raise ValueError("The original checkpoint contains no model.visual tensors")
    output.mkdir(parents=True, exist_ok=False)
    tensors, records = {}, {}
    for shard in sorted({weight_map[name] for name in names}):
        if Path(shard).name != shard:
            raise ValueError("Expected a local checkpoint shard basename")
        with safe_open(source / shard, framework="pt", device="cpu") as file:
            for name in (name for name in names if weight_map[name] == shard):
                tensor = file.get_tensor(name).contiguous()
                tensors[name] = tensor
                records[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                 "bytes": tensor.numel() * tensor.element_size(),
                                 "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy()).hexdigest(),
                                 "source_shard": shard}
    weights = output / "vision.safetensors"
    save_file(tensors, weights, metadata={"format": "pt", "component": "original unquantized vision encoder and merger",
                                        "source_model": args.model_id, "source_revision": source.name})
    assets = ("config.json", "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "chat_template.jinja",
              "generation_config.json")
    copied = {}
    for name in assets:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, output / name)
            copied[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if not any(name in copied for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError("Original image processor config is required")
    receipt = {"source_model": args.model_id, "source_revision": source.name,
               "component": "All original model.visual tensors, including the vision encoder and merger/projector",
               "quantized": False, "dtype_conversion": False, "tensors": records,
               "tensor_bytes": sum(record["bytes"] for record in records.values()),
               "weights_file": weights.name, "weights_bytes": weights.stat().st_size,
               "weights_sha256": file_sha256(weights), "assets": copied,
               "validation": "Extracted tensor byte hashes recorded; image inference is a separate release check"}
    (output / "vision-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("source_model", "source_revision", "quantized", "tensor_bytes",
                                                  "weights_bytes", "weights_sha256")}))
    print("Preserved", len(records), "original vision tensors and", len(copied), "processor/tokenizer assets")


if __name__ == "__main__":
    main()
