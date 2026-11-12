# W-ModelForge

[中文说明](README.zh-CN.md)

W-ModelForge is [W-World-Labs](https://github.com/W-World-Labs)' experimental project for model compression and inference acceleration. We maintain practical adaptations, reproducible recipes and measured results covering artifact size, quality, build cost and inference behavior.

The first integrated method is **LookAtMyQuant**. The project uses a direct method directory and working scripts, without a plugin framework.

| Method | Code and source | Current scope |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/), from [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), MIT | Qwen3.5-9B / Qwen3.8-27B language PTQ: trellis coding, LDLQ, Hadamard processing, Fisher row allocation and upstream EMA codebooks, with CPU/GPU offload and shared decomposition reuse. |

- [Inference guide](inference/README.md): download/checksum and multimodal reference launch scripts, dense HF export, and optional vLLM / llama.cpp conversion and serving. External-engine quality and speed remain unmeasured.
- [Colab workflow, Chinese](colab/README.md): Google's official hosted Colab MCP, reproducible execution, rclone copy and verification.
- [Upstream and changes](UPSTREAM.md): fixed source baseline, downstream changes and maintenance rules.
- [Third-party notices](NOTICE.md) / [license](LICENSE): separate terms for code, model weights and datasets.

Models are published under [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs). Release requires file integrity, independent packed loading and actual inference, at least 90% relative correct-answer retention on the selected 216-problem math protocol, and PPL/KL review. Model cards disclose results, revisions, licenses and loading limits.

## Current results

As of 2026-11-12, [Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2) is public. Its language packed directory is **3.438 GB**. At a 1024-token generation cap, original / packed scores are **83 / 80 correct out of 216**, or **96.39% relative retention**; **131 / 133** responses hit the cap. This length-limited synthetic-arithmetic result does not establish broad capability retention. Additional short-window PPL rises from **22.81 to 32.07**; primary WikiText KL is **0.331**. See the model card for full results and limitations.

Only language weights are quantized. The original **BF16 visual encoder and merger/projector** are preserved: all 333 visual tensors from each source model have been extracted unchanged and backed up. The complete public 9B release is approximately **4.374 GB**, including original vision and processor assets. Actual loading matches every visual tensor's original byte hash, answers `red` / `blue` for two basic images, and matches the dense quantized language reference (KL approximately 1.205e-11, identical top-1 predictions and four short greedy prompts).

Qwen3.8-27B language packing is complete: **9.643 GB across 8 files**, with independent Drive backup verified without differences. Matched original / packed math evaluation at 1024 tokens and subsequent multimodal validation are pending; **27B remains private**.

The recipe is PTQ without distillation or QAT. Language weights use the upstream custom packed format, rather than GGUF or standard Transformers weights. The reference loader expands weights into a dense model; **packed download size is not inference VRAM usage**. No general inference-speed improvement is established. Coding, tool use and video quality are unbenchmarked. Upstream 4B results are attributed background, not W-ModelForge results.

## Attribution

This repository retains LookAtMyQuant's Git history and GitHub fork relationship for traceability. **W-ModelForge's broader scope includes other compression and acceleration methods.** The original compiler, recipes, cached codebooks and historical experiments belong to Shin Megami Boson. Our changes are listed in [UPSTREAM.md](UPSTREAM.md); no endorsement by the original author or Qwen is implied.
