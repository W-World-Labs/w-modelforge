# Download, inference and backend conversion

[中文说明](README.zh-CN.md)

Use the complete model release: `packed/`, original BF16 `vision.safetensors`, processor assets, `evaluation.json` and checksums. Bash scripts support Linux/macOS; the HF repositories include matching copies.

| Path | Input | Validation scope |
| --- | --- | --- |
| `run_reference.sh` | Custom packed language + BF16 vision | Reuses the actually validated `load_packed.py`; expands into a dense Torch model |
| `export_hf.py` → `serve_vllm.sh` | Expanded standard HF safetensors | Optional export/launch path; no vLLM quality or speed measurement |
| `export_hf.py` → `convert_llamacpp.sh` → `serve_llamacpp.sh` | Standard HF → language GGUF + mmproj GGUF | Optional conversion/launch path; no llama.cpp quality or speed measurement |

**vLLM and llama.cpp cannot directly load `opgraph-packed-1`.** Optional backend paths expand weights and lose the current packed storage size. These scripts do not implement native trellis kernels. Further Q4/IQ or other GGUF quantization is a new lossy conversion requiring separate evaluation; existing scores do not transfer to it.

## Download and run the reference model

Install hardware-appropriate PyTorch in an isolated environment first. The validated environment uses Torch 2.14.1+cu130 / Transformers 5.16.1. Keep external inference engines in separate environments from active Colab quantization.

```bash
git clone https://github.com/W-World-Labs/w-modelforge.git
cd w-modelforge
python -m pip install transformers==5.16.1 numpy==2.4.6 huggingface_hub safetensors pillow
bash inference/download.sh W-World-Labs/Qwen3.5-9B-LAMQ2 ./models/qwen35-9b
# An optional third argument pins an HF commit; default: main.
# Anonymous 27B downloads require its release review to pass and the repo to become public:
# bash inference/download.sh W-World-Labs/Qwen3.8-27B-LAMQ2 ./models/qwen38-27b

bash inference/run_reference.sh ./models/qwen35-9b \
  --prompt 'Hello, introduce yourself in one sentence.' --max-new-tokens 128
bash inference/run_reference.sh ./models/qwen35-9b \
  --image ./models/qwen35-9b/examples/vision-red-square.png \
  --prompt 'Name the color of the large shape. Answer with one color word.'
```

Download verifies size and SHA256 for every packed file and original vision. The reference launcher reads build dtype from `evaluation.json`: FP32 for 9B, BF16 for 27B. The loader retains complete processor output. Set `DEVICE=cpu` for slower CPU inference and FP64 decoding.

When running HF script copies, set `W_MODELFORGE_ROOT=/absolute/path/to/w-modelforge` to locate the code checkout; pass the model release directory first.

Expanded parameter storage is approximately **36.7 GB for 9B** (FP32 language + BF16 vision) and **54.7 GB for 27B** (BF16 language + vision), plus decoding temporaries, KV cache and framework overhead. Download size does not describe inference memory.

## Export standard Hugging Face weights

```bash
python methods/lookatmyquant/export_hf.py ./models/qwen35-9b ./expanded/qwen35-9b \
  --device cuda --decode-device cuda
```

The output directory must not exist. Export reuses packed decoding and preserves language build dtype, original BF16 visual parameters and processor assets. It writes sharded safetensors, full configuration and `w-modelforge-export.json` provenance. Shards target 4 GB; the model moves to CPU for saving, requiring sufficient host RAM. Allow space for input, expanded output and temporaries: exported parameters are approximately 36.7 GB for 9B or 54.7 GB for 27B. These are **dense reconstructions of quantized weights**, not newly downloaded original language weights.

Dtypes above describe safetensors storage. Ordinary HF reloads or external engines may convert parameters to one execution dtype. Export records language dtype in root/text configuration and BF16 in vision configuration; check execution precision separately.

## vLLM serving

In another environment, follow [official installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) and use a version supporting the Qwen3.5 architecture:

```bash
bash inference/serve_vllm.sh ./expanded/qwen35-9b
# For 27B: bash inference/serve_vllm.sh ./expanded/qwen38-27b
```

Defaults: `127.0.0.1:8000`, context 4096, served name `w-modelforge`, one image per request. Append extra flags. Environment overrides: `HOST`, `PORT`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `SERVED_MODEL_NAME`.

Default `DTYPE=bfloat16` follows the current Qwen GDN CUDA fast path. **9B FP32 language weights are narrowed to BF16**, outside its validated FP32 configuration; evaluate separately. 27B is already BF16. Architecture support alone does not validate these exports in vLLM.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"w-modelforge","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

Image requests use OpenAI-compatible `messages[].content` with `text` and `image_url`, including base64 data URLs. See [multimodal inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/).

## llama.cpp conversion and serving

The statically reviewed commit is [`93cf736a8d1d46aa922e27eb64aa712bcd5ebc30`](https://github.com/ggml-org/llama.cpp/tree/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30). Review later upstream changes and follow the [official build guide](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/docs/build.md).

The converter has its own environment: the pinned upstream requirements request Torch 2.11 and Transformers 4.57.6. Keep the reference export environment above for `export_hf.py`, then use the converter environment only for GGUF conversion. Colab images without ensurepip can replace the environment-creation line with `uv venv --seed --python python .venv-gguf`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 93cf736a8d1d46aa922e27eb64aa712bcd5ebc30
python -m venv .venv-gguf
.venv-gguf/bin/python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# NVIDIA CUDA:
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# Apple Metal: omit -DGGML_CUDA=ON above; Metal is enabled by default.
cmake --build llama.cpp/build --config Release -j 8

PATH="$PWD/.venv-gguf/bin:$PATH" \
  bash inference/convert_llamacpp.sh ./expanded/qwen35-9b ./llama.cpp ./gguf/qwen35-9b
LLAMA_SERVER=./llama.cpp/build/bin/llama-server \
  bash inference/serve_llamacpp.sh ./gguf/qwen35-9b/language.gguf ./gguf/qwen35-9b/mmproj.gguf
```

Language conversion uses `--no-mtp`; the vision converter excludes MTP itself and does not accept this flag. Language output follows build dtype: F32 for 9B, BF16 for 27B, avoiding `auto` silently selecting F16 for 9B. mmproj uses BF16 with converter-required F32 tensors, without extra low-bit visual quantization. Visual convolutions split and language attention tensors reorder, so **GGUF layouts/bytes differ from the original 333-tensor safetensors layout**. Original BF16 vision remains in the downloaded release. Receipts record engine commit, sizes and SHA256.

Environment overrides: `LLAMA_SERVER`, `N_GPU_LAYERS`, `HOST`, `PORT`, `MAX_MODEL_LEN`, `SERVED_MODEL_NAME`; append extra llama-server flags. Multimodal serving requires both language GGUF and mmproj. See the [official multimodal guide](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/docs/multimodal.md).

## Longer math216 comparisons

`compare_math_caps.py` repeats the same frozen synthetic-arithmetic protocol. Current profiles are 27B BF16/batch8 at independently generated 2048 and 4096 caps, and 9B FP32/batch4 at 4096. Original and packed models run separately, one GPU model per process; PPL/parity/vision are not rerun. Existing 1024 reports remain unchanged.

```bash
python methods/lookatmyquant/compare_math_caps.py --kind original --model-id Qwen/Qwen3.8-27B --model-path /path/to/original27-snapshot --dtype bfloat16 --batch 8 --max-new 2048 4096 --output ./runs/math-caps27
python methods/lookatmyquant/compare_math_caps.py --kind packed --model-id Qwen/Qwen3.8-27B --packed ./models/qwen38-27b/packed --dtype bfloat16 --batch 8 --max-new 2048 4096 --output ./runs/math-caps27
# For 9B, use its original snapshot/packed path, --model-id Qwen/Qwen3.5-9B,
# --dtype float32 --batch 4 --max-new 4096 and a separate output directory.
```

Reports contain full generated text, scores and truncations. They are atomically saved after each complete batch; rerunning identical arguments resumes only when protocol, model, source, assets and runtime fingerprints match. Optional `--rclone-remote mydrive:project/math-caps27` copies/checks only original.json, packed.json and comparison.json after each batch. Use a dedicated remote subdirectory; logs/locks stay outside the output directory. One explicitly allowlisted JSON-validation-only driver repair is compatible with the initial `3e08e36` driver; model assets and generation/scoring/loader hashes stay exact, and each report retains its real initial driver fingerprint. These are quality comparisons, not isolated throughput benchmarks.

## Sources and licenses

Scripts and `export_hf.py`: W-World-Labs, MIT; packed decoding: Shin Megami Boson's LookAtMyQuant, MIT. See [NOTICE](https://github.com/W-World-Labs/w-modelforge/blob/main/NOTICE.md). Qwen weights retain Apache-2.0. vLLM (Apache-2.0) and llama.cpp (MIT) are external dependencies, not vendored source.

Primary references: [vLLM support](https://docs.vllm.ai/en/stable/models/supported_models/), [serving flags](https://docs.vllm.ai/en/stable/cli/serve/), [Qwen GDN](https://github.com/vllm-project/vllm/blob/9e6815acab1a6acb188931510271b1bce4688c75/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L942), [Qwen conversion](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/conversion/qwen.py#L694), [visual conversion](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/conversion/qwen3vl.py#L16). These describe upstream architecture/format behavior, not measurements for this release.
