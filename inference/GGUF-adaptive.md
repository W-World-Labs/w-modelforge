# Adaptive standard GGUF

[中文说明](GGUF-adaptive.zh-CN.md)

This path reuses upstream LookAtMyQuant's `gguf_compiler.py` and `compiler.py` allocator. It measures each original BF16 model tensor's output KL under standard llama.cpp Q/IQ encodings, then minimizes the summed measured loss under a storage budget. Allocation is offline and per tensor. This is a fresh quantization of the original model, separate from the row-wise trellis artifacts and their dense GGUF exports. Results must be measured independently; neither existing LAMQ scores nor native trellis support is implied.

Use the validated Torch/Transformers reference environment for measurement, and a separate converter environment as described in [the inference guide](README.md). Build llama.cpp at `93cf736a8d1d46aa922e27eb64aa712bcd5ebc30`, including `llama-imatrix`, `llama-quantize`, `llama-perplexity` and `llama-server`. The Qwen3.5 GDN inverse layout follows that pinned converter; incomplete tensor mappings or unexpected stored types abort the build. Original BF16 vision remains separate and is not quantized again.

The current pilot uses six 512-token sensitivity windows, 32 calibration chunks and the following candidate menu. Those are measurement/calibration inputs, separate from the frozen math216 acceptance set. The initial byte budget matches a stock Q3_K_M file; the final file size is checked separately. This development recipe is not a completed quality claim.

```bash
export PYTHONPATH=/absolute/path/llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH}
GGUF_WORK=/absolute/path/gguf-adaptive9
METHOD=methods/lookatmyquant
GGUF_ARGS=(--model Qwen/Qwen3.5-9B
  --model-path /absolute/path/to/pinned-original-snapshot
  --work-dir "$GGUF_WORK" --llama-dir /absolute/path/llama.cpp
  --converter-python /absolute/path/converter-env/bin/python
  --menu Q2_K,Q3_K,IQ4_XS,Q4_K,Q6_K,BF16 --stock Q3_K_M
  --calib 32 --context 512 --eval-chunks 8 --dtype bfloat16)

python "$METHOD/gguf_compiler.py" prepare "${GGUF_ARGS[@]}"
python "$METHOD/gguf_compiler.py" measure "${GGUF_ARGS[@]}" \
  --windows 6 --kl-batch 2 --out "$GGUF_WORK/sensitivity.json"

# The allocator budgets controlled tensors; reserve the base file's fixed
# tensors, metadata/alignment and 4096 bytes inside the stock file size.
GGUF_BUDGET_BITS=$(python - "$GGUF_WORK" <<'PY'
import json, sys
from pathlib import Path
from gguf import GGUFReader
w = Path(sys.argv[1])
controlled = set(json.loads((w / 'gguf_map.json').read_text())['map'])
base = GGUFReader(str(w / 'bf16.gguf'))
fixed = sum(int(t.n_bytes) for t in base.tensors if t.name not in controlled)
overhead = (w / 'bf16.gguf').stat().st_size - sum(int(t.n_bytes) for t in base.tensors)
print(8 * ((w / 'stock_Q3_K_M.gguf').stat().st_size - fixed - overhead - 4096))
PY
)
python "$METHOD/compiler.py" allocate --sens "$GGUF_WORK/sensitivity.json" \
  --target disk --items linear,embed,head --budget "$GGUF_BUDGET_BITS" \
  --out "$GGUF_WORK/adaptive.json"
python "$METHOD/gguf_compiler.py" build "${GGUF_ARGS[@]}" \
  --sens "$GGUF_WORK/sensitivity.json" --plan "$GGUF_WORK/adaptive.json" --default Q4_K
python "$METHOD/gguf_compiler.py" evaluate "${GGUF_ARGS[@]}" \
  --files stock_Q3_K_M.gguf,plan_adaptive.gguf --out "$GGUF_WORK/native-quality.json"
```

The global build type must be quantized: this llama.cpp version skips per-tensor quantized overrides when the global type is BF16/F16/F32. Explicit overrides still preserve the plan's individual BF16 tensors. The builder verifies every planned stored type and byte count.

For 27B use its exact original snapshot, `Qwen/Qwen3.8-27B`, and a separate work directory. Temporary BF16/candidate files and native KL base logits require substantial additional disk; allocate space before starting. Keep active source revisions fixed and do not deploy changes into running evaluations.

Run native math separately for the original BF16 GGUF and the adaptive GGUF. `math_gguf.py` renders prompts with the original tokenizer, checks all 216 native tokenizations, then submits the same token IDs with greedy decoding, four slots, 8192 context per slot, a 4096 generation cap, F16 KV and prompt caching disabled. It saves raw generated token IDs and complete text after each cohort. Both models receive the same two basic image prompts. This is an independent native-engine protocol, rather than an assignment of Torch scores to GGUF.

```bash
python "$METHOD/math_gguf.py" --kind original --model-id Qwen/Qwen3.5-9B \
  --snapshot /absolute/path/to/pinned-original-snapshot --gguf "$GGUF_WORK/bf16.gguf" \
  --mmproj /absolute/path/mmproj.gguf --llama-dir /absolute/path/llama.cpp \
  --images /absolute/path/to/published/examples --out "$GGUF_WORK/math-original.json" --port 18081
python "$METHOD/math_gguf.py" --kind adaptive --model-id Qwen/Qwen3.5-9B \
  --snapshot /absolute/path/to/pinned-original-snapshot --gguf "$GGUF_WORK/plan_adaptive.gguf" \
  --mmproj /absolute/path/mmproj.gguf --llama-dir /absolute/path/llama.cpp \
  --images /absolute/path/to/published/examples --out "$GGUF_WORK/math-adaptive.json" --port 18082
```

After a model's GGUF package is published under `gguf-adaptive/`, `MODEL_FORMAT=gguf bash inference/download.sh REPOSITORY ./models/native` downloads only that package and verifies its checksums. The default packed download excludes the optional GGUF files. Use `serve_llamacpp.sh` with the downloaded language GGUF and matching mmproj.

Native perplexity/KL, independent math216, full-file checksums and basic image inference must accompany publication. Provide the matching BF16/F32 mmproj and state any converter-required F32 visual tensors. Standard llama.cpp compatibility does not establish compatibility with every version of a downstream application or prove a throughput improvement.

Source/compiler code: Shin Megami Boson and W-World-Labs, MIT; external llama.cpp, MIT; Qwen weights, Apache-2.0. See [NOTICE](../NOTICE.md) and [UPSTREAM](../UPSTREAM.md).

## Independent 9B results

Original BF16 → adaptive GGUF: math 183→184/216 at cap 4096, cap hits 29→24, relative count 100.55%, on the same native four-slot/F16-KV profile. Both models passed the two basic images. Language size 4,622,312,672 bytes; original BF16/F32 mmproj adds 921,704,384 bytes. Eight 512-context chunks: WikiText PPL 8.915811→9.263493, KL 0.038616; mixed-text PPL 5.393594→5.448359, KL 0.041946. Stock Q3_K_M PPL is 9.476296/5.737011 and KL 0.076973/0.067423. These short measurements are separate from LAMQ. Binomial falls 8→6/18; the single-run result does not establish a capability gain or general speedup. Full raw reports and bilingual cards accompany the independent GGUF package. The 27B run is still measuring.
