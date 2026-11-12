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

27B 使用对应固定原始快照、`Qwen/Qwen3.8-27B` 和独立工作目录。BF16 中间文件、多个候选及原生 KL 参考 logits 占用较大磁盘，应预留空间；运行中的量化与评测保持固定源码。

公开时必须附带匹配的 BF16/F32 mmproj、SHA256、实际大小、配方、独立数学/PPL/KL结果，以及转换器要求的视觉 F32 张量说明。标准 llama.cpp 可加载不等于所有下游应用版本均兼容，也不代表已证明速度提升。

上游编译器与维护代码为 MIT，外部 llama.cpp 为 MIT，Qwen 权重为 Apache-2.0；详见 [NOTICE](../NOTICE.md) 与 [UPSTREAM](../UPSTREAM.md)。
