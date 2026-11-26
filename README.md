# W-ModelForge

[中文说明](README.zh-CN.md)

W-ModelForge is [W-World-Labs](https://github.com/W-World-Labs)' independently maintained project for model compression and inference acceleration. We maintain practical adaptations, reproducible recipes and measured results covering artifact size, quality, build cost and inference behavior.

**LookAtMyQuant** is the first upstream dependency. Shin Megami Boson maintains it independently. W-ModelForge owns the Colab recipes, inference tooling, evaluation and release workflows, and the adaptations that connect them to the upstream code.

## Source setup

```sh
git clone --recurse-submodules https://github.com/W-World-Labs/w-modelforge.git
cd w-modelforge
bash scripts/prepare_lookatmyquant.sh
```

[`upstream/lookatmyquant`](upstream/lookatmyquant) is the upstream Git submodule, pinned to LookAtMyQuant 0.2.0 (`e36d2141015c26074fd0450e08729522c87c0775`). The preparation script applies the W-maintained [compatibility patch](patches/lookatmyquant.patch) in a generated `methods/lookatmyquant/` directory used by the existing recipes and scripts. We review upstream updates and commit the new pin and corresponding patch together. [Upstream maintenance and changes](UPSTREAM.md)

The Colab recipe defaults to the historical W-ModelForge source revision for reproducibility. Pass `--source-commit <full-w-modelforge-sha>` with a new run ID to use the current layout. [Colab workflow](colab/README.md)

| Method | Code and source | Current scope |
| --- | --- | --- |
| LookAtMyQuant | Generated at `methods/lookatmyquant/` from [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), MIT | Qwen3.5-9B / Qwen3.8-27B language PTQ: trellis coding, LDLQ, Hadamard processing, Fisher row allocation and upstream EMA codebooks, with CPU/GPU offload and shared decomposition reuse. |
| Adaptive standard GGUF | Upstream `gguf_compiler.py` + `compiler.py`, MIT; [recipe](inference/GGUF-adaptive.md) | Fresh original-BF16 quantization with measured per-tensor KL and size-budget allocation over standard llama.cpp Q/IQ types. 9B and 27B native releases are public, with independently reviewed quality results below. |

- [Inference guide](inference/README.md): download/checksum and multimodal reference launch scripts, dense HF export, and optional vLLM / llama.cpp conversion and serving. Legacy export quality and speed remain unmeasured; fresh-original GGUF has independent quality results below.
- [Colab workflow, Chinese](colab/README.md): Google's official hosted Colab MCP, reproducible execution, rclone copy and verification.
- [Upstream and changes](UPSTREAM.md): fixed source baseline, downstream changes and maintenance rules.
- [Third-party notices](NOTICE.md) / [license](LICENSE): separate terms for code, model weights and datasets.

## Source validation

[GitHub Actions](https://github.com/W-World-Labs/w-modelforge/actions/workflows/source.yml) runs source validation for each formal GitHub Release. It scans the complete public Git history and checkout with checksum-pinned Gitleaks 8.30.1, limits every historical blob to 10 MiB and the current source tree to 100 MiB, rejects model/data caches and engine binaries, and checks Python/JSON/Bash syntax plus credential-free Colab CLI help. Only the four unchanged upstream EMA codebooks are permitted as tensor files. No runtime dependencies are installed and no models, datasets, credentials or release artifacts are uploaded by CI.

These are static source checks, not model-quality, GPU-performance, production or commercial-rights acceptance. The inherited unit/integration tests are retained for provenance and are not run. Local checkouts are for editing and metadata/secret review; CI validation runs on GitHub.

Models are published under [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs). Release requires file integrity, independent format-specific loading and actual inference, at least 90% relative correct-answer retention on the selected 216-problem math protocol, and PPL/KL review. Model cards disclose results, revisions, licenses and loading limits.

## Current results

As of 2026-11-13 (Asia/Shanghai), [Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2) is public, with **3.438 GB** of packed language files. At the current **4096-token** cap, original / actual packed scores are **188 / 175 out of 216** (**93.09% relative correct-answer count**), with **24 / 29** cap hits. The same frozen synthetic arithmetic protocol uses FP32/batch4. Historical 1024 results remain **83 / 80**, **96.39%**, and **131 / 133** cap hits. These are restricted, single-run results; family-level losses vary. Additional packed PPL rises **22.81→32.07 (+40.60%)**; main quantizer-dense WikiText KL is **0.331**. Full answers, source/recovery records and limitations are in the model card.

Only language weights are quantized. The original **BF16 visual encoder and merger/projector** are preserved: all 333 visual tensors from each source model have been extracted unchanged and backed up. The packed-format 9B download, excluding optional GGUF, is approximately **4.38 GB**, including original vision and processor assets. Actual loading matches every visual tensor's original byte hash, answers `red` / `blue` for two basic images, and matches the dense quantized language reference (KL approximately 1.205e-11, identical top-1 predictions and four short greedy prompts).

[Qwen3.8-27B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.8-27B-LAMQ2) is now public, with **9.643 GB** of packed language files plus **0.9215 GB** of original BF16 vision. At the current **4096-token** cap, original / actual packed scores are **194 / 183 out of 216**, or **94.33% relative correct-answer count**, with **17 / 26** cap hits. At 2048 the counts are **160 / 155 (96.88%)**, with **55 / 58** cap hits; historical 1024 remains 84 / 86. All four longer-cap generations were independently rescored. The BF16/batch8 protocol and full answers are in the bilingual model card. Loss is material and uneven: binomial falls 10→5 out of 18 at4096, additional packed PPL rises48.04%, and dense-to-packed reconstruction remains non-exact (KL 0.000691, top-1 agreement98.63%). All333 original BF16 visual hashes and two basic image examples pass.

The published LAMQ recipe is PTQ without distillation or QAT. Its language weights use the upstream custom packed format, rather than GGUF or standard Transformers weights. The reference loader expands weights into a dense model; **packed download size is not inference VRAM usage**. No general inference-speed improvement is established. Coding, tool use and video quality are unbenchmarked. Upstream 4B results are attributed background, not W-ModelForge results.

The provided 9B CPU standard-HF export preserves all 333 original visual byte hashes. F32 language GGUF + BF16/F32 mmproj conversion and the llama.cpp CUDA launcher were actually run, answering `red`, `blue` and `156` in three basic requests. These are startup/format checks for that LAMQ export; its engine math/PPL/speed remain unmeasured, and vLLM execution remains unvalidated.


The separate fresh-original **adaptive standard GGUF** pilot has built a **4.622 GB** 9B language file within the stock Q3_K_M size budget. On eight 512-context native WikiText chunks, BF16/stock/adaptive PPL is 8.916/9.476/9.263, and stock/adaptive KL is 0.07697/0.03862. Independent native math 4096 completed at **183/216 BF16 → 184/216 adaptive (100.55% relative count)**, with **29→24** cap hits. All 432 rows were rescored and decoded by the frozen driver; both models passed two basic image prompts. These restricted single-run results are separate from LAMQ; one additional correct answer does not establish a capability gain. Family losses remain (binomial 8→6/18). The original native BF16 baseline differs from the historical Torch FP32 baseline 188/216. Vision mmproj adds 0.922 GB of unquantized BF16/F32 data. The [9B native package](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2/tree/main/gguf-adaptive) is public with separate bilingual cards and raw reports. Its actual anonymous CLI/checksum flow and downloaded launch script passed red/blue/156 requests; large files were reused from the hash-verified release, with independent anonymous remote-header checks. The 27B GGUF native review also passed: **193→191/216 (98.96%)** at cap 4096, with **20→19** cap hits. Its language file is **13.287 GB**, plus **0.931 GB** unquantized vision, within the 13.301 GB stock Q3_K_M language budget. WikiText BF16/stock/adaptive PPL is 6.836698/7.135067/7.037839, with stock/adaptive KL 0.052455/0.040364. Mixed-text PPL is 4.707599/4.811088/4.815769 and KL 0.044038/0.030154; stock has slightly lower mixed PPL. Native math used two existing G4 instances with byte-identical baselines, engine libraries and CUDA libraries and matching recorded profiles. All 432 rows passed rescoring/token-decoder checks; both models passed both images. Addition/subtraction falls 17→15/18, so family losses remain. The [27B native package](https://huggingface.co/W-World-Labs/Qwen3.8-27B-LAMQ2/tree/main/gguf-adaptive) is public. Its actual anonymous CLI/checksum flow and published startup script passed red/blue/156 requests, using preseeded hash-verified large files and independent anonymous remote-header checks. Full compiler/helper MIT text is supplied separately from the weights Apache-2.0 license. These results do not replace legacy LAMQ scores. See the [independent recipe and evaluation protocol](inference/GGUF-adaptive.md).

## Attribution

LookAtMyQuant's original compiler, recipes, fixed codebooks and historical experiments are credited to Shin Megami Boson. The submodule records their upstream source and revision. [Upstream and changes](UPSTREAM.md) identifies the adaptations maintained by W-World-Labs. Attribution does not imply endorsement by the original author or Qwen.
