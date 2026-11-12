# W-ModelForge

[中文说明](README.zh-CN.md)

W-ModelForge is [W-World-Labs](https://github.com/W-World-Labs)' experimental project for model compression and inference acceleration. We maintain practical adaptations, reproducible recipes and measured results covering artifact size, quality, build cost and inference behavior.

The first integrated method is **LookAtMyQuant**. The project uses a direct method directory and working scripts, without a plugin framework.

| Method | Code and source | Current scope |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/), from [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), MIT | Qwen3.5-9B / Qwen3.8-27B language PTQ: trellis coding, LDLQ, Hadamard processing, Fisher row allocation and upstream EMA codebooks, with CPU/GPU offload and shared decomposition reuse. |
| Adaptive standard GGUF | Upstream `gguf_compiler.py` + `compiler.py`, MIT; [recipe](inference/GGUF-adaptive.md) | Fresh original-BF16 quantization with measured per-tensor KL and size-budget allocation over standard llama.cpp Q/IQ types. Colab/Qwen adaptation is in progress; independent quality results are pending. |

- [Inference guide](inference/README.md): download/checksum and multimodal reference launch scripts, dense HF export, and optional vLLM / llama.cpp conversion and serving. External-engine quality and speed remain unmeasured.
- [Colab workflow, Chinese](colab/README.md): Google's official hosted Colab MCP, reproducible execution, rclone copy and verification.
- [Upstream and changes](UPSTREAM.md): fixed source baseline, downstream changes and maintenance rules.
- [Third-party notices](NOTICE.md) / [license](LICENSE): separate terms for code, model weights and datasets.

Models are published under [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs). Release requires file integrity, independent packed loading and actual inference, at least 90% relative correct-answer retention on the selected 216-problem math protocol, and PPL/KL review. Model cards disclose results, revisions, licenses and loading limits.

## Current results

As of 2026-11-13 (Asia/Shanghai), [Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2) is public, with **3.438 GB** of packed language files. At the current **4096-token** cap, original / actual packed scores are **188 / 175 out of 216** (**93.09% relative correct-answer count**), with **24 / 29** cap hits. The same frozen synthetic arithmetic protocol uses FP32/batch4. Historical 1024 results remain **83 / 80**, **96.39%**, and **131 / 133** cap hits. These are restricted, single-run results; family-level losses vary. Additional packed PPL rises **22.81→32.07 (+40.60%)**; main quantizer-dense WikiText KL is **0.331**. Full answers, source/recovery records and limitations are in the model card.

Only language weights are quantized. The original **BF16 visual encoder and merger/projector** are preserved: all 333 visual tensors from each source model have been extracted unchanged and backed up. The complete public 9B release is approximately **4.38 GB**, including original vision and processor assets. Actual loading matches every visual tensor's original byte hash, answers `red` / `blue` for two basic images, and matches the dense quantized language reference (KL approximately 1.205e-11, identical top-1 predictions and four short greedy prompts).

[Qwen3.8-27B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.8-27B-LAMQ2) is now public, with **9.643 GB** of packed language files plus **0.9215 GB** of original BF16 vision. At the current **4096-token** cap, original / actual packed scores are **194 / 183 out of 216**, or **94.33% relative correct-answer count**, with **17 / 26** cap hits. At 2048 the counts are **160 / 155 (96.88%)**, with **55 / 58** cap hits; historical 1024 remains 84 / 86. All four longer-cap generations were independently rescored. The BF16/batch8 protocol and full answers are in the bilingual model card. Loss is material and uneven: binomial falls 10→5 out of 18 at4096, additional packed PPL rises48.04%, and dense-to-packed reconstruction remains non-exact (KL0.000691, top-1 agreement98.63%). All333 original BF16 visual hashes and two basic image examples pass.

The published LAMQ recipe is PTQ without distillation or QAT. Its language weights use the upstream custom packed format, rather than GGUF or standard Transformers weights. The reference loader expands weights into a dense model; **packed download size is not inference VRAM usage**. No general inference-speed improvement is established. Coding, tool use and video quality are unbenchmarked. Upstream 4B results are attributed background, not W-ModelForge results.

The provided 9B CPU standard-HF export preserves all 333 original visual byte hashes. F32 language GGUF + BF16/F32 mmproj conversion and the llama.cpp CUDA launcher were actually run, answering `red`, `blue` and `156` in three basic requests. These are startup/format checks for that LAMQ export; its engine math/PPL/speed remain unmeasured, and vLLM execution remains unvalidated.


The separate fresh-original **adaptive standard GGUF** pilot has built a **4.622 GB** 9B language file within the stock Q3_K_M size budget. On eight512-context native WikiText chunks, BF16/stock/adaptive PPL is8.916/9.476/9.263, and stock/adaptive KL is0.07697/0.03862. These are short native-engine measurements, separate from LAMQ scores; original/adaptive math4096 is running before public GGUF release. The27B GGUF adaptation is also in progress. See the [independent recipe and evaluation protocol](inference/GGUF-adaptive.md).

## Attribution

This repository retains LookAtMyQuant's Git history and GitHub fork relationship for traceability. **W-ModelForge's broader scope includes other compression and acceleration methods.** The original compiler, recipes, cached codebooks and historical experiments belong to Shin Megami Boson. Our changes are listed in [UPSTREAM.md](UPSTREAM.md); no endorsement by the original author or Qwen is implied.
