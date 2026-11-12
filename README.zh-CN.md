# W-ModelForge

[English](README.md)
W-ModelForge 是 [W-World-Labs](https://github.com/W-World-Labs) 的模型量化与加速实验项目。我们在这里维护具体方案的适配、运行配方和实测结果，逐步比较压缩大小、质量、量化成本与推理表现。

当前只有一种已接入的方法：**LookAtMyQuant**。先把实际方案跑通并记录结果，不提前设计插件框架或通用后端接口。

| 方案 | 代码与来源 | 当前范围 |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/)，来自 [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant)，MIT | Qwen3.5-9B / Qwen3.8-27B 文本 PTQ；trellis、LDLQ、Hadamard、Fisher 行级分配和上游 EMA 码本；CPU/GPU 卸载及重复分解复用。 |

- [`colab/`](colab/README.md)：通过 Google 官方托管 Colab MCP 管理 G4，执行配方，使用 rclone 回传并校验。
- [`inference/`](inference/README.zh-CN.md)：下载与校验、完整多模态参考启动，以及标准 HF 导出、vLLM 和 llama.cpp 转换/启动脚本；后两者的后端质量和速度尚未实测。
- [来源与改动](UPSTREAM.md)：固定源码基线、我们的修改和维护约定。
- [版权与第三方说明](NOTICE.md) / [许可证](LICENSE)：代码、模型权重与数据的许可分别记录。

权重发布组织为 [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs)。只有文件完整、packed 权重能够独立加载并完成实际推理、216题数学成绩保留原模型至少90%，且PPL/KL无明显异常的版本才公开；评测结果、模型来源、许可与加载范围随模型卡一起发布。

## 当前进度

2026-11-12：9B已公开于[W-World-Labs/Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2)，packed实测3.438GB。当前4096上限、FP32/batch4、相同冻结216题，原始/实际packed为**188/175，相对正确数93.09%**，24/29题达到上限。历史1024仍保留83/80、96.39%、131/133截断。各子任务损失并不均匀，不能视作综合能力保留率；额外packed PPL为22.81→32.07（+40.60%），主WikiText的dense KL约0.331。完整回答、源码与恢复记录见模型卡。

3.8-27B语言packed已完成，实测9.643 GB、8文件，HF私有副本及网盘备份均已核验。1024-token数学原始/packed为84/86题（各216题），131/130题达到长度上限；333个原始BF16视觉哈希及两次实际图像推理通过。packed不是dense精确重建：额外短窗口两者KL约0.000691、top-1一致98.63%，packed PPL比原始上升48.04%。27B原始BF16/batch8的2048/4096已完成160/194题，55/17题达到上限，独立packed对照正在运行，1024报告全部保留；27B继续私有等待扩展复核。9B完整多模态版约4.38 GB，已公开且保留原始BF16视觉。尚无总体推理加速结论。

当前配方只量化语言权重，不含蒸馏；视觉组件保持原始BF16。语言产物为原方案自定义 packed 格式，不是 GGUF 或标准 Transformers 权重；两模型的视觉哈希及基础图像推理已核验，广泛视觉、编码、工具调用和视频质量尚未评估。上游的4B成绩只在方法文档中作为来源背景保留，不能当作 W-ModelForge 的结果。

9B标准HF导出已实际完成，333个视觉张量保持原始BF16哈希；F32语言GGUF与BF16/F32 mmproj转换、llama.cpp CUDA启动也已实际执行，三次基础请求回答red、blue、156。这只核验格式/启动，尚无该后端数学/PPL/速度基准，vLLM仍未执行。

## 来源

这个仓库保留了 LookAtMyQuant 的 Git 提交历史和 GitHub fork 关系，以便追溯首个方法的来源；**W-ModelForge 的项目范围包括后续其他量化与加速方案**。LookAtMyQuant 编译器、原始配方、缓存码本和历史实验归原作者 Shin Megami Boson。我们的修改另列于 [UPSTREAM.md](UPSTREAM.md)，不表示上游作者或 Qwen 的背书。
