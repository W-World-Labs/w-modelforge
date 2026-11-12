# 下载、启动与后端转换

这些脚本适用于本项目发布的完整模型目录：`packed/`、原始 BF16 `vision.safetensors`、processor、`evaluation.json` 和校验清单。Shell 脚本可在 Linux/macOS 的 Bash 中运行。复制到 Hugging Face 的同名脚本与此处保持一致。

| 路径 | 输入 | 当前验证范围 |
| --- | --- | --- |
| `run_reference.sh` | 原始自定义 packed + BF16 视觉 | 复用已实际验证的 `load_packed.py`，展开为密集 Torch 权重 |
| `export_hf.py` → `serve_vllm.sh` | 展开后的标准 HF safetensors | 可选转换/启动入口；不代表已完成 vLLM 质量或速度评测 |
| `export_hf.py` → `convert_llamacpp.sh` → `serve_llamacpp.sh` | 标准 HF → language GGUF + mmproj GGUF | 可选转换/启动入口；不代表已完成 llama.cpp 质量或速度评测 |

**vLLM 和 llama.cpp 不能直接读取本项目的 `opgraph-packed-1`。** 后两条路径会展开权重，失去当前 packed 文件的低比特存储体积；脚本没有实现 trellis 原生推理内核。再次用 GGUF Q4/IQ 等量化属于新的有损转换，需要重新评估，不能沿用当前模型卡分数。

## 下载并启动已验证的参考实现

先克隆 W-ModelForge，在独立环境中安装与你的 CUDA 匹配的 PyTorch。已验证环境为 Torch 2.14.1+cu130 / Transformers 5.16.1；不要为安装另一个推理引擎覆盖正在量化的 Colab 环境。

```bash
git clone https://github.com/W-World-Labs/w-modelforge.git
cd w-modelforge
python -m pip install transformers==5.16.1 numpy==2.4.6 huggingface_hub safetensors pillow

bash inference/download.sh W-World-Labs/Qwen3.5-9B-LAMQ2 ./models/qwen35-9b
# 第三个位置参数可固定某次 HF commit；省略时下载 main。
# 27B 只有在质量评估通过并公开后，才能匿名下载：
# bash inference/download.sh W-World-Labs/Qwen3.8-27B-LAMQ2 ./models/qwen38-27b

bash inference/run_reference.sh ./models/qwen35-9b \
  --prompt '你好，请用一句话介绍自己。' --max-new-tokens 128
bash inference/run_reference.sh ./models/qwen35-9b \
  --image ./models/qwen35-9b/examples/vision-red-square.png \
  --prompt 'Name the color of the large shape. Answer with one color word.'
```

下载后自动核对所有 packed 文件及原始视觉文件的大小与 SHA256。启动脚本从 `evaluation.json` 选择实际构建 dtype：9B 为 FP32，27B 为 BF16；完整 processor 输出由既有 loader 保留。CPU 路径可设置 `DEVICE=cpu`，默认同时在 CPU 做 FP64 解码，速度较慢。

若直接使用 HF 仓库内的脚本，在命令前设置 `W_MODELFORGE_ROOT=/绝对路径/w-modelforge`，让脚本找到代码库。模型根目录仍作为第一个参数传入。

9B 的参考参数存储约 **36.7 GB**（FP32 语言 + BF16 视觉）；27B 的 BF16 语言加视觉约 **54.7 GB**，均另需解码临时内存、KV cache 和框架开销。这是展开后的存储预算，不能用下载文件大小推断运行显存。

## 导出标准 Hugging Face 权重

```bash
python methods/lookatmyquant/export_hf.py ./models/qwen35-9b ./expanded/qwen35-9b \
  --device cuda --decode-device cuda
```

目标目录必须尚不存在。导出复用相同的 packed 解码，保留语言构建 dtype、全部原始 BF16 视觉参数和 processor，输出分片 safetensors、完整配置与 `w-modelforge-export.json` 来源记录。默认每片 4 GB；保存时模型移回 CPU，因此也需要足够主机内存。9B 导出约 36.7 GB；27B 约 54.7 GB，请为输入、输出和临时文件同时留空间。这里的权重是**量化结果的密集重建**，不是重新下载的原始语言模型。

上述 dtype 指 safetensors 中的实际存储。普通 HF 重载或第三方引擎可能按统一执行 dtype 再转换参数；导出会将根配置与 text_config 写为语言构建 dtype，vision_config 标为 BF16，实际执行精度仍须按所用引擎核对。

## vLLM 服务

在另一个独立环境中按 [vLLM 官方安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) 安装支持 Qwen3.5 架构的版本，然后运行：

```bash
bash inference/serve_vllm.sh ./expanded/qwen35-9b
# 27B 对应：bash inference/serve_vllm.sh ./expanded/qwen38-27b
```

默认监听 `127.0.0.1:8000`、上下文 4096、服务名 `w-modelforge`，允许每个请求一张图片；额外 vLLM 参数直接追加。可用 `HOST`、`PORT`、`MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`、`SERVED_MODEL_NAME` 环境变量调整。

默认 `DTYPE=bfloat16`，符合当前 Qwen GDN CUDA 快路径。**9B 的 FP32 语言权重会在 vLLM 中转为 BF16**，其输出不属于模型卡上已验证的 FP32 配置；需要单独评估。27B 的构建 dtype 已是 BF16。不能仅凭架构在支持列表中就认定本次导出权重已经通过 vLLM 评测。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"w-modelforge","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

图像请求按 OpenAI 兼容接口，在 `messages[].content` 中放置 `text` 和 `image_url`（可用图片的 base64 data URL）；具体参数见 [vLLM 多模态输入说明](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)。

## llama.cpp 转换与服务

静态核对的 llama.cpp 版本为 [`93cf736a8d1d46aa922e27eb64aa712bcd5ebc30`](https://github.com/ggml-org/llama.cpp/tree/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30)，后续版本需按其上游变更检查。先按 [官方构建说明](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/docs/build.md) 编译。

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 93cf736a8d1d46aa922e27eb64aa712bcd5ebc30
python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# NVIDIA CUDA：
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# Apple Metal：上面一行省略 -DGGML_CUDA=ON，Metal 默认启用。
cmake --build llama.cpp/build --config Release -j 8

bash inference/convert_llamacpp.sh ./expanded/qwen35-9b ./llama.cpp ./gguf/qwen35-9b
LLAMA_SERVER=./llama.cpp/build/bin/llama-server \
  bash inference/serve_llamacpp.sh ./gguf/qwen35-9b/language.gguf ./gguf/qwen35-9b/mmproj.gguf
```

语言转换明确使用 `--no-mtp`；视觉转换器自行忽略 MTP，不接收该标志。语言 GGUF 按导出 dtype 选择：9B 使用 F32，27B 使用 BF16，避免 `auto` 将 9B 默默转换为 F16；mmproj 使用 BF16，并保留转换器要求的 F32 张量，没有额外低比特视觉量化。Qwen 视觉卷积会拆分、语言注意力也会重排，因此 **GGUF 的布局和张量字节不是原始 333 张量的 safetensors 布局**；原始 BF16 文件仍保留在下载目录。转换记录包含引擎 commit、输出大小和 SHA256。

运行参数可通过 `LLAMA_SERVER`、`N_GPU_LAYERS`、`HOST`、`PORT`、`MAX_MODEL_LEN`、`SERVED_MODEL_NAME` 调整，额外 llama-server 参数直接追加。多模态必须同时传语言 GGUF 和 mmproj；官方接口见 [llama.cpp 多模态说明](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/docs/multimodal.md)。

## 来源与许可

脚本和 `export_hf.py` 为 W-World-Labs 的 MIT 代码，复用 Shin Megami Boson / LookAtMyQuant 的 MIT packed 解码，详见仓库 [NOTICE](https://github.com/W-World-Labs/w-modelforge/blob/main/NOTICE.md)。模型沿用 Qwen 权重的 Apache-2.0 许可。vLLM（Apache-2.0）和 llama.cpp（MIT）是外部依赖，没有将其源码并入本项目。

后端依据：[vLLM 支持列表](https://docs.vllm.ai/en/stable/models/supported_models/)、[服务参数](https://docs.vllm.ai/en/stable/cli/serve/)、[Qwen3.5 GDN 实现](https://github.com/vllm-project/vllm/blob/9e6815acab1a6acb188931510271b1bce4688c75/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L942)、[llama.cpp Qwen 转换](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/conversion/qwen.py#L694)、[视觉转换](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/conversion/qwen3vl.py#L16)。这些是上游格式和架构支持的证据，不是本项目的后端实测报告。
