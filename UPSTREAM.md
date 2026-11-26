# Upstream and downstream changes

## Provenance

- Original project: [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), by Shin Megami Boson.
- Original upstream baseline for the historical W-ModelForge results: [`4716ffee3da205a8d1df94e4abc629040893fa53`](https://github.com/ShinMegamiBoson/lookatmyquant/commit/4716ffee3da205a8d1df94e4abc629040893fa53). The individual run receipts record their exact W-ModelForge and model revisions.
- Current submodule pin: LookAtMyQuant 0.2.0, [`e36d2141015c26074fd0450e08729522c87c0775`](https://github.com/ShinMegamiBoson/lookatmyquant/commit/e36d2141015c26074fd0450e08729522c87c0775), dated 2026-11-20.
- License: MIT. The original copyright and permission text remain in [LICENSE](LICENSE).
- Integrating project: [W-World-Labs/w-modelforge](https://github.com/W-World-Labs/w-modelforge). The pristine upstream checkout is the Git submodule `upstream/lookatmyquant`. W-World-Labs maintains its changes in [`patches/lookatmyquant.patch`](patches/lookatmyquant.patch).

## W-World-Labs changes, 2026-11-12

Run `git submodule update --init` followed by `bash scripts/prepare_lookatmyquant.sh` to generate the adapted working directory. Source filenames below are relative to `methods/lookatmyquant/`; W-ModelForge also maintains its own `colab/`, `inference/` and project documents.

| Files | Change |
| --- | --- |
| `quantize.py` | CPU offload of reference and quantized models; release saved CPU tensor references before offloaded evaluation; stage current decoder layers; use a single resident model for repeated statistics; chunked GPU trellis state recovery; reuse the same input rotation and read-only environment within a stage. |
| `opgraph/codes.py` | Bounded, complete-row CUDA lookup and path validation; reuse prepared input rotations and Hessian environments without changing row-dependent sweep state. |
| `math_calibration.py`, `row_importance.py`, `math_eval.py` | Explicit FP32/BF16 options, preserving FP32 defaults; model-specific evaluation cache tags. |
| `compiler_artifact.py` | Carry source model license and notice assets when available. |
| `load_packed.py`, `opgraph/hadamard.py` | Standalone Torch dense reference loading of the packed artifact, including row groups, vocabulary buckets, aliases and BF16 state. Optional original BF16 vision/merger weights restore the full Qwen3.5 multimodal wrapper without a second language model; nonpersistent rotary buffers retain their initialized precision. The model initializes directly on its requested device before per-matrix decoding, preserving real rotary buffers while avoiding a full CPU initialization beside another running model. Existing Hadamard rules moved unchanged into a model-config-independent module shared with `opgraph/codes.py`; no packed inference kernel is added. |
| `extract_vision.py` | Preserve all original BF16 visual encoder/merger tensors and processor assets, with per-tensor byte hashes and source revision; no quantization or dtype conversion. |
| `export_hf.py`, root `inference/` | Downstream download/checksum and reference launch scripts; dense HF export reuses the same packed decoder and preserves original BF16 vision. Optional vLLM and llama.cpp conversion/launch scripts use external engines, not native packed inference, and do not establish backend quality or speed. No external engine source is vendored. |
| `compare_math_caps.py` | Incremental longer-generation comparisons on the unchanged math216 protocol, reusing upstream generation/scoring and the existing loaders. Independent caps, complete-batch resume with source/model fingerprints, full answer text, and optional rclone copy/check of three JSON reports; no new benchmark or scoring rule. |
| `math_gguf.py` | Independent native llama.cpp original/adaptive GGUF evaluation on the same frozen math216 problems and existing exact-answer scorer. Verifies HF/native prompt token IDs, records greedy raw output tokens, F16 KV, fixed four-slot cohorts, basic image responses and artifact/source fingerprints; resumes complete cohorts without assigning Torch scores to GGUF. |
| `gguf_compiler.py`, `public_eval.py`, `compiler.py` | Reuse the upstream GGUF compiler and allocator for standard llama.cpp tensor types without importing an unrelated trellis model. Add pinned local snapshots, separate converter Python, configurable candidate/calibration/evaluation sizes, incremental sensitivity reports, strict planned-type/size verification, complete native PPL/base-PPL/KL logs and input hashes, and inverse Qwen3.5 GDN V-head ordering only for matching matrix shapes, with exact original-BF16 mapping checks. The inverse ordering follows llama.cpp `93cf736a8d1d46aa922e27eb64aa712bcd5ebc30`, `conversion/qwen.py` (MIT); no external source is vendored. This is fresh quantization from original BF16, separate from trellis reconstruction/export. |
| `release_validation.py` | Two-process actual-model release evaluation: fixed small held-out logprob/PPL comparison, public prompt generation and packed-model math216 using the existing scorer; reports hashes and math retention without uploading or approving publication. |
| `colab/` | Official hosted Colab MCP client and a parameterized 9B/27B PTQ/rclone workflow; no credentials, model weights, or private run logs are committed. |
| `README.md`, `NOTICE.md`, `LICENSE`, `UPSTREAM.md` | Project scope, upstream credit, license separation and experiment limits. |

The trellis path retains the upstream trellis/LDLQ/Hadamard recipe with its cached EMA codebooks. Precision of the FP64 quantization algebra, CPU table ordering, nearest-level tie rules and path checks is retained. The current experiments do not include the upstream 4,000-step distillation.

## Maintenance

The 0.2.0 pin adds upstream Python packaging, a `lamq` CLI entrypoint, and Apple Silicon MLX runtime source. The reconstructed W-ModelForge compiler, quantization, and evaluation paths and all four EMA codebooks match the prior distribution byte for byte. Existing published measurements retain their recorded source revisions. The upstream CLI and MLX runtime still need execution checks.

On 2026-11-13, downstream source maintenance added GitHub-only static CI: complete-history/tree secret scanning, source-size and distribution-scope gates, Python/JSON/Bash syntax and credential-free Colab CLI entrypoints. It preserves the four original fixed EMA codebooks by exact Git blob, clarifies optional dependency licenses, and aligns the Colab README with the recipe's existing source pin. No quantization implementation, model artifact or measured result changed; CI is not model/GPU or commercial acceptance.

Fetch new revisions from the submodule remote and review changes since the recorded pin, including the MIT notice, fixed codebooks and the APIs used by W-ModelForge. Refresh the compatibility patch against the reviewed commit and stage the new gitlink with `git add upstream/lookatmyquant`. Move any previously prepared `methods/lookatmyquant/` work tree aside, then run `bash scripts/prepare_lookatmyquant.sh` and validate the affected entrypoints. Commit the changed gitlink, its matching patch and updated provenance in one W-ModelForge commit.

Record the W-ModelForge commit, upstream commit, model revision, dtype, recipe, environment and output hashes for each new run. Reproduce existing runs from their recorded source commits.

The initial trellis builds used the previously deployed9B memory patch, with later source optimizations applied to the queued27B process; those builds are complete. A running process is not automatically updated by a Git push. Completed-layer counts and GPU readings are progress observations, not reproducibility or quality claims.

The separate 9B adaptive standard-GGUF run completed its native release review: original BF16 183/216 versus adaptive 184/216 at cap4096, with 29/24 cap hits; short WikiText PPL 8.915811/9.263493 and KL 0.038616. These results use frozen measurement/build/math revisions recorded with the artifact, rather than replacing legacy LAMQ scores. The original-BF16 27B GGUF run also passed review:193→191/216,20→19 cap hits,98.9637% relative count, with unchanged cross-runtime original report and matching full native/engine/CUDA fingerprints. Its mixed PPL is slightly above stock and is disclosed.

Both reviewed native GGUF packages are public with separate raw results and offline MIT helper-license text alongside Apache-2.0 model licenses. Actual anonymous helper/checksum flows and published native launch scripts passed red/blue/156 basic requests, with large-file reuse disclosed in their receipts; no formal throughput or broad capability claim is added.
