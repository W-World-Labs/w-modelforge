# W-ModelForge

W-ModelForge 是 [W-World-Labs](https://github.com/W-World-Labs) 的模型量化与加速实验项目。我们在这里维护具体方案的适配、运行配方和实测结果，逐步比较压缩大小、质量、量化成本与推理表现。

当前只有一种已接入的方法：**LookAtMyQuant**。先把实际方案跑通并记录结果，不提前设计插件框架或通用后端接口。

| 方案 | 代码与来源 | 当前范围 |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/)，来自 [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant)，MIT | Qwen3.5-9B / Qwen3.8-27B 文本 PTQ；trellis、LDLQ、Hadamard、Fisher 行级分配和上游 EMA 码本；CPU/GPU 卸载及重复分解复用。 |

- [`colab/`](colab/README.md)：通过 Google 官方托管 Colab MCP 管理 G4，执行配方，使用 rclone 回传并校验。
- [来源与改动](UPSTREAM.md)：固定源码基线、我们的修改和维护约定。
- [版权与第三方说明](NOTICE.md) / [许可证](LICENSE)：代码、模型权重与数据的许可分别记录。

## 当前进度

2026-11-12：9B 已生成校准解答并完成 Fisher 分配，正在逐层量化；3.8-27B 已下载并准备环境，排在9B之后。尚无 W-ModelForge 完整模型的压缩大小、质量保留率或总体加速结论。显存使用随阶段变化，当前采样不能代表整个任务峰值。

当前配方只做文本 PTQ，不含蒸馏。产物为原方案自定义 packed 格式，不是 GGUF 或标准 Transformers 权重；图像/视频能力、编码和工具调用尚未评估。上游的4B成绩只在方法文档中作为来源背景保留，不能当作 W-ModelForge 的结果。

## 来源

这个仓库保留了 LookAtMyQuant 的 Git 提交历史和 GitHub fork 关系，以便追溯首个方法的来源；**W-ModelForge 的项目范围包括后续其他量化与加速方案**。LookAtMyQuant 编译器、原始配方、缓存码本和历史实验归原作者 Shin Megami Boson。我们的修改另列于 [UPSTREAM.md](UPSTREAM.md)，不表示上游作者或 Qwen 的背书。
