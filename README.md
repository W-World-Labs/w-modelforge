# W-ModelForge

W-ModelForge 是 [W-World-Labs](https://github.com/W-World-Labs) 的模型量化与加速实验项目。我们在这里维护具体方案的适配、运行配方和实测结果，逐步比较压缩大小、质量、量化成本与推理表现。

当前只有一种已接入的方法：**LookAtMyQuant**。先把实际方案跑通并记录结果，不提前设计插件框架或通用后端接口。

| 方案 | 代码与来源 | 当前范围 |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/)，来自 [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant)，MIT | Qwen3.5-9B / Qwen3.8-27B 文本 PTQ；trellis、LDLQ、Hadamard、Fisher 行级分配和上游 EMA 码本；CPU/GPU 卸载及重复分解复用。 |

- [`colab/`](colab/README.md)：通过 Google 官方托管 Colab MCP 管理 G4，执行配方，使用 rclone 回传并校验。
- [来源与改动](UPSTREAM.md)：固定源码基线、我们的修改和维护约定。
- [版权与第三方说明](NOTICE.md) / [许可证](LICENSE)：代码、模型权重与数据的许可分别记录。

权重发布组织为 [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs)。只有文件完整、packed 权重能够独立加载并完成实际推理、216题数学成绩保留原模型至少90%，且PPL/KL无明显异常的版本才公开；评测结果、模型来源、许可与加载范围随模型卡一起发布。

## 当前进度

2026-11-12：9B 语言权重已完成量化、独立加载和评估，并发布到 [W-World-Labs/Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2)。packed 目录实测3.438 GB；1024-token上限下数学216题原始83、packed80（相对保留96.39%），分别131/133题被截断，不能视为广泛能力保留率。额外短窗口PPL为22.81→32.07，主WikiText KL为0.331；完整结果与限制见模型卡。

3.8-27B正在构建。按当前需求，语言权重量化，视觉编码器和merger/projector保留原始BF16：两模型各333个视觉张量已原样提取，正在补齐多模态加载与图文验证。当前9B发布大小仅含语言packed，追加视觉后的总大小另报；尚无总体推理加速结论。

当前配方只量化语言权重，不含蒸馏；视觉组件保持原始BF16。语言产物为原方案自定义 packed 格式，不是 GGUF 或标准 Transformers 权重；多模态路径正在验证，编码、工具调用和视频质量尚未评估。上游的4B成绩只在方法文档中作为来源背景保留，不能当作 W-ModelForge 的结果。

## 来源

这个仓库保留了 LookAtMyQuant 的 Git 提交历史和 GitHub fork 关系，以便追溯首个方法的来源；**W-ModelForge 的项目范围包括后续其他量化与加速方案**。LookAtMyQuant 编译器、原始配方、缓存码本和历史实验归原作者 Shin Megami Boson。我们的修改另列于 [UPSTREAM.md](UPSTREAM.md)，不表示上游作者或 Qwen 的背书。
