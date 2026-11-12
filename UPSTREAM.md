# Upstream and downstream changes

## Provenance

- Original project: [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), by Shin Megami Boson.
- Baseline: [`4716ffee3da205a8d1df94e4abc629040893fa53`](https://github.com/ShinMegamiBoson/lookatmyquant/commit/4716ffee3da205a8d1df94e4abc629040893fa53).
- License: MIT. The original copyright and permission text remain in [LICENSE](LICENSE).
- Maintained fork: [W-World-Labs/w-modelforge](https://github.com/W-World-Labs/w-modelforge). GitHub's fork relationship and original commit history are retained.

## W-World-Labs changes, 2026-11-12

The inherited source has moved to `methods/lookatmyquant/`. Source filenames below are relative to that directory; `colab/` and project documents remain at the repository root.

| Files | Change |
| --- | --- |
| `quantize.py` | CPU offload of reference and quantized models; release saved CPU tensor references before offloaded evaluation; stage current decoder layers; use a single resident model for repeated statistics; chunked GPU trellis state recovery; reuse the same input rotation and read-only environment within a stage. |
| `opgraph/codes.py` | Bounded, complete-row CUDA lookup and path validation; reuse prepared input rotations and Hessian environments without changing row-dependent sweep state. |
| `math_calibration.py`, `row_importance.py`, `math_eval.py` | Explicit FP32/BF16 options, preserving FP32 defaults; model-specific evaluation cache tags. |
| `compiler_artifact.py` | Carry source model license and notice assets when available. |
| `load_packed.py`, `opgraph/hadamard.py` | Standalone Torch dense reference loading of the packed artifact, including row groups, vocabulary buckets, aliases and BF16 state. Optional original BF16 vision/merger weights restore the full Qwen3.5 multimodal wrapper without a second language model; nonpersistent rotary buffers retain their initialized precision. The model initializes directly on its requested device before per-matrix decoding, preserving real rotary buffers while avoiding a full CPU initialization beside another running model. Existing Hadamard rules moved unchanged into a model-config-independent module shared with `opgraph/codes.py`; no packed inference kernel is added. |
| `extract_vision.py` | Preserve all original BF16 visual encoder/merger tensors and processor assets, with per-tensor byte hashes and source revision; no quantization or dtype conversion. |
| `export_hf.py`, root `inference/` | Downstream download/checksum and reference launch scripts; dense HF export reuses the same packed decoder and preserves original BF16 vision. Optional vLLM and llama.cpp conversion/launch scripts use external engines, not native packed inference, and do not establish backend quality or speed. No external engine source is vendored. |
| `release_validation.py` | Two-process actual-model release evaluation: fixed small held-out logprob/PPL comparison, public prompt generation and packed-model math216 using the existing scorer; reports hashes and math retention without uploading or approving publication. |
| `colab/` | Official hosted Colab MCP client and a parameterized 9B/27B PTQ/rclone workflow; no credentials, model weights, or private run logs are committed. |
| `README.md`, `NOTICE.md`, `LICENSE`, `UPSTREAM.md` | Fork identity, upstream credit, license separation and experiment limits. |

The algorithm remains the upstream trellis/LDLQ/Hadamard recipe with its cached EMA codebooks. Precision of the FP64 quantization algebra, CPU table ordering, nearest-level tie rules and path checks is retained. The current experiments do not include the upstream 4,000-step distillation.

## Maintenance

Use `upstream` for the original repository and `origin` for this fork. Keep downstream commits separate from the baseline; inspect future upstream changes before merging. Record the source commit, model revision, dtype, recipe, environment and output hashes for each run. Never substitute an unrecorded branch head into an existing run.

The active 9B process uses its previously deployed memory patch; later source optimizations first apply to the queued 27B process. A running process is not automatically updated by a Git push. Completed-layer counts and GPU readings are progress observations, not reproducibility or quality claims.
