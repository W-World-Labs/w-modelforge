# 标准 GGUF 自适应混合精度

[English guide](GGUF-adaptive.md)

复用上游 LookAtMyQuant 的 `gguf_compiler.py` 与 `compiler.py` 分配器：测量原始 BF16 模型各张量在标准 llama.cpp Q/IQ 编码下造成的输出 KL，再在存储预算内选择预计总损失较小的组合。这是离线、按张量的分配，直接从原模型重新量化；与此前按行 trellis 编码及其展开导出分别记录。既有 LAMQ 分数不能沿用，也不代表 llama.cpp 支持原生 trellis。

测量使用已验证的 Torch/Transformers 环境；转换器使用[推理指南](README.zh-CN.md)中的独立环境。固定 llama.cpp 为 `93cf736a8d1d46aa922e27eb64aa712bcd5ebc30`，构建 `llama-imatrix`、`llama-quantize`、`llama-perplexity` 和 `llama-server`。Qwen3.5 GDN 的逆排列对应此版本转换器；张量映射不完整或实际存储类型偏离计划时终止。原始 BF16 视觉单独保留，不再次低比特量化。

首轮使用六个 512-token 敏感度窗口、32 个校准块，以及英文指南所列 Q2_K/Q3_K/IQ4_XS/Q4_K/Q6_K/BF16 菜单。校准和选码窗口与冻结 math216 验收题分开。以标准 Q3_K_M 的实际大小作为首轮预算参照；固定张量、元数据和对齐另外计入最终文件大小。当前是开发配方，尚无完成的质量结论。

[完整命令见英文指南](GGUF-adaptive.md)。依次执行：

1. `prepare`：原始 BF16 GGUF、重要性矩阵、候选编码和标准 Q3_K_M 对照。
2. `measure`：逐张量替换，测量相对原模型的 KL；保留输入、源码和实际存储类型记录。
3. `compiler.py allocate`：在扣除固定开销的字节预算内选择类型。
4. `build`：用标准 `llama-quantize --tensor-type` 构建，核对实际类型与存储字节。
5. `evaluate`：通过原生 llama.cpp 测量 PPL/KL；随后独立完成原始/自适应 GGUF 的数学评测和图像核验。

构建全局类型使用 `--default Q4_K`；此版本 llama.cpp 在全局为 BF16/F16/F32 时会跳过逐张量低比特覆盖。计划中需要保留的 BF16 张量仍通过明确覆盖保留，构建后逐项核对。

`math_gguf.py` 分别评估原始 BF16 GGUF 和自适应 GGUF：用原始 tokenizer 渲染提示词，先逐题核对 216 份原生分词，再提交相同 token ID，使用贪心解码、四个槽位、每槽 8192 上下文、4096 生成上限和 F16 KV，关闭提示词缓存。每组保存生成 token ID 与全文，两种模型均运行相同的两例图像提示。它与此前 Torch 成绩分别记录；命令见英文指南。

模型通过评估、在 `gguf-adaptive/` 发布后，使用 `MODEL_FORMAT=gguf bash inference/download.sh 仓库 ./models/native` 仅下载并校验原生 GGUF 包。默认 packed 下载会排除可选 GGUF 文件；用 `serve_llamacpp.sh` 传入语言文件及匹配的 mmproj 即可启动。

27B 使用对应固定原始快照、`Qwen/Qwen3.8-27B` 和独立工作目录。BF16 中间文件、多个候选及原生 KL 参考 logits 占用较大磁盘，应预留空间；运行中的量化与评测保持固定源码。

公开时必须附带匹配的 BF16/F32 mmproj、SHA256、实际大小、配方、独立数学/PPL/KL结果，以及转换器要求的视觉 F32 张量说明。标准 llama.cpp 可加载不等于所有下游应用版本均兼容，也不代表已证明速度提升。

上游编译器与维护代码为 MIT，外部 llama.cpp 为 MIT，Qwen 权重为 Apache-2.0；详见 [NOTICE](../NOTICE.md) 与 [UPSTREAM](../UPSTREAM.md)。

## 9B 独立结果

原始 BF16 → 自适应 GGUF：4096 上限下数学183→184/216，截断29→24，相对正确数100.55%；同一原生四路/F16 KV协议，双方两张基础图片均通过。语言4,622,312,672字节，原始BF16/F32 mmproj另921,704,384字节。八个512上下文块：WikiText PPL8.915811→9.263493、KL0.038616；混合文本PPL5.393594→5.448359、KL0.041946。标准Q3_K_M对应PPL9.476296/5.737011、KL0.076973/0.067423。此短评测与旧LAMQ结果分开；组合数8→6/18，小样本单次结果不证明能力提升或总体加速。完整原始记录、英文模型卡及中文文件随独立GGUF包发布。27B结果见下方。

[已审核的9B包](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2/tree/main/gguf-adaptive)现已公开。敏感度实际取mix校准分区（seed1），原生mix评估取独立eval分区（seed2）；原始报告沿用的 `held-out windows` 字段记录尺寸，不代表独立测试集。公开模型卡提供实际下载/启动回执，原始回答哈希保持不变。

## 27B 独立结果

语言文件13,287,312,192字节，低于标准Q3_K_M的13,301,443,392字节；未量化BF16/F32视觉另931,145,888字节。原生4096数学193→191/216，相对98.9637%，截断20→19；同一四路、每路8192上下文、F16 KV配置。原始/自适应分别用既有两台G4并行，GPU/driver相同，原基线GGUF、mmproj、引擎及五份CUDA库同字节。432条回答重打分与冻结驱动解码均过，双方red/blue两图通过；原始答案原样导入第二台，仅验证完成态。

每输入8个512上下文块：WikiText BF16/标准/自适应PPL6.836698/7.135067/7.037839，标准/自适应KL0.052455/0.040364；混合PPL4.707599/4.811088/4.815769，KL0.044038/0.030154。标准混合PPL略低，自适应两项KL更低。加减17→15/18，单次小样本不代表综合能力保留或总体加速。测量/构建/数学源码均4eb5f2fee030ab4e1582b2fb515e2daa5a6655fa；代码MIT、权重Apache-2.0，正在准备公开。
