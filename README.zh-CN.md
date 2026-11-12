# W-ModelForge

[English](README.md)
W-ModelForge 是 [W-World-Labs](https://github.com/W-World-Labs) 的模型量化与加速实验项目。我们在这里维护具体方案的适配、运行配方和实测结果，逐步比较压缩大小、质量、量化成本与推理表现。

当前维护 **LookAtMyQuant** trellis 流程，并适配其标准 GGUF 编译路径。先把实际方案跑通并记录结果，不提前设计插件框架或通用后端接口。

| 方案 | 代码与来源 | 当前范围 |
| --- | --- | --- |
| LookAtMyQuant | [`methods/lookatmyquant/`](methods/lookatmyquant/)，来自 [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant)，MIT | Qwen3.5-9B / Qwen3.8-27B 文本 PTQ；trellis、LDLQ、Hadamard、Fisher 行级分配和上游 EMA 码本；CPU/GPU 卸载及重复分解复用。 |
| 标准 GGUF 自适应量化 | 上游 `gguf_compiler.py` 与 `compiler.py`，MIT；[配方](inference/GGUF-adaptive.zh-CN.md) | 从原始 BF16 重新量化，对标准 Q/IQ 类型实测张量 KL、按大小预算分配。9B与27B原生版本均已公开，独立评测成绩分别记录。 |

- [`colab/`](colab/README.md)：通过 Google 官方托管 Colab MCP 管理 G4，执行配方，使用 rclone 回传并校验。
- [`inference/`](inference/README.zh-CN.md)：下载与校验、完整多模态参考启动，以及标准 HF 导出、vLLM 和 llama.cpp 转换/启动脚本；LAMQ展开导出的后端质量/速度仍未测；原BF16生成的GGUF另有独立原生成绩。
- [来源与改动](UPSTREAM.md)：固定源码基线、我们的修改和维护约定。
- [版权与第三方说明](NOTICE.md) / [许可证](LICENSE)：代码、模型权重与数据的许可分别记录。

权重发布组织为 [Hugging Face / W-World-Labs](https://huggingface.co/W-World-Labs)。只有文件完整、对应格式权重能够独立加载并完成实际推理、216题数学成绩保留原模型至少90%，且PPL/KL无明显异常的版本才公开；评测结果、模型来源、许可与加载范围随模型卡一起发布。

## 当前进度

2026-11-13（北京时间）：9B已公开于[W-World-Labs/Qwen3.5-9B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2)，packed实测3.438GB。当前4096上限、FP32/batch4、相同冻结216题，原始/实际packed为**188/175，相对正确数93.09%**，24/29题达到上限。历史1024仍保留83/80、96.39%、131/133截断。各子任务损失并不均匀，不能视作综合能力保留率；额外packed PPL为22.81→32.07（+40.60%），主WikiText的dense KL约0.331。完整回答、源码与恢复记录见模型卡。

[Qwen3.8-27B-LAMQ2](https://huggingface.co/W-World-Labs/Qwen3.8-27B-LAMQ2)现已公开：语言packed为 **9.643 GB**，原始BF16视觉另占 **0.9215 GB**。当前4096上限、BF16/batch8，原始/实际packed为 **194/183题，相对94.33%**，截断 **17/26**；2048档 **160/155，相对96.88%**，截断55/58；历史1024为84/86。四组长输出结果均已逐题独立重打分，全文见双语模型卡。损失明显且分布不均：4096档组合数10→5/18；额外packed PPL上升48.04%；dense/packed重建并非精确一致（KL约0.000691，top-1一致98.63%）。333个原始BF16视觉哈希和两例基础图像均通过；尚无总体推理加速结论。

已发布LAMQ配方只量化语言权重，不含蒸馏；视觉组件保持原始BF16。语言产物为原方案自定义 packed 格式，不是 GGUF 或标准 Transformers 权重；两模型的视觉哈希及基础图像推理已核验，广泛视觉、编码、工具调用和视频质量尚未评估。上游的4B成绩只在方法文档中作为来源背景保留，不能当作 W-ModelForge 的结果。

9B标准HF导出已实际完成，333个视觉张量保持原始BF16哈希；F32语言GGUF与BF16/F32 mmproj转换、llama.cpp CUDA启动也已实际执行，三次基础请求回答red、blue、156。这只核验该LAMQ展开导出的格式/启动，尚无它的原生数学/PPL/速度基准，vLLM仍未执行。


另行从原始BF16生成的 **标准GGUF自适应方案**，已构建9B语言文件 **4.622 GB**，在标准Q3_K_M大小预算内。原生8个512上下文WikiText块中，BF16/标准/自适应PPL为8.916/9.476/9.263，标准/自适应KL为0.07697/0.03862。独立原生4096数学已完成：**BF16 183/216→自适应184/216，相对正确数100.55%**，截断 **29→24**。432条回答已逐题重打分，并由冻结驱动重新解码核验；双方两例基础图像均通过。小样本单次多对一题不能证明能力提升，组合数仍8→6/18。原生BF16基线也不同于历史Torch FP32的188分。未量化BF16/F32视觉mmproj另占0.922GB。这些结果与LAMQ分开记录；[9B GGUF包](https://huggingface.co/W-World-Labs/Qwen3.5-9B-LAMQ2/tree/main/gguf-adaptive)已公开，附双语说明和原始报告。实际匿名CLI/哈希流程及公开包启动脚本已通过red/blue/156请求；大文件复用已验哈希的本次权重，匿名远端HEAD另核身份。27B GGUF也已通过原生评测审核：4096上限 **193→191/216，相对98.96%**，截断 **20→19**；语言 **13.287 GB**、未量化视觉另 **0.931 GB**，在13.301GB标准Q3_K_M语言预算内。WikiText的BF16/标准/自适应PPL为6.836698/7.135067/7.037839，标准/自适应KL为0.052455/0.040364；混合PPL为4.707599/4.811088/4.815769，KL为0.044038/0.030154，标准混合PPL略低。数学使用既有两台G4并行，基线文件、引擎/共享库及CUDA库完全同字节，记录的配置一致；432条回答已重打分/解码复核，双方两图均通过。加减仍17→15/18，不能视为综合能力保留率；[27B原生包](https://huggingface.co/W-World-Labs/Qwen3.8-27B-LAMQ2/tree/main/gguf-adaptive)现已公开。实际匿名CLI/哈希流程及公开启动脚本通过red/blue/156请求；大文件复用已验SHA的本次权重，匿名远端HEAD另核身份。完整代码MIT与模型Apache-2.0分别提供，见[独立配方与评估流程](inference/GGUF-adaptive.zh-CN.md)。

## 来源

这个仓库保留了 LookAtMyQuant 的 Git 提交历史和 GitHub fork 关系，以便追溯首个方法的来源；**W-ModelForge 的项目范围包括后续其他量化与加速方案**。LookAtMyQuant 编译器、原始配方、缓存码本和历史实验归原作者 Shin Megami Boson。我们的修改另列于 [UPSTREAM.md](UPSTREAM.md)，不表示上游作者或 Qwen 的背书。
