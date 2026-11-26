# Maintenance instructions

- 用最简单、直接、改动最少的实现，优先复用现有文件、函数和类型。
- 仅为当前实际需求或明显重复逻辑增加抽象，不为未来扩展提前设计。
- 替换实现时移除旧路径，避免新旧双轨。
- 禁止写单元测试和集成测试；只做静态检查、源代码审阅和实际模型质量评估。
- W-ModelForge 维护项目配方、推理工具和评估流程。LookAtMyQuant 由上游独立维护，源码通过 `upstream/lookatmyquant` Git 子模块固定在当前审阅的提交，保留其提交历史、版权与许可证。当前固定版本、原始实验基线及各轮更新说明见 UPSTREAM.md。
- 用 `git submodule update --init` 和 `bash scripts/prepare_lookatmyquant.sh` 准备适配后的 `methods/lookatmyquant/`。持久改动写入 `patches/lookatmyquant.patch`。审阅上游变更后同步提交 gitlink、对应补丁与 UPSTREAM.md/NOTICE.md。
- 量化进程使用固定源码和模型版本；不重启正在运行的任务来部署可选优化。
- 不提交 OAuth、rclone 配置、私有日志、权重或数据缓存。
- 安装、检查与构建在 GitHub CI 执行；本机只编辑源码并核对元数据、体积与凭据。正式 GitHub Release 触发的 `Source checks` 仅做静态语法、CLI help、完整公开历史/树脱敏和源码范围门禁，不下载模型或执行现有测试。四个上游固定 EMA 码本按精确 Git blob 保留；CI 不产生模型/数据/引擎发布产物，不代表模型质量、GPU 性能或商用权利验收。
- 在给用户回复的最后一段文字中，用一两百个字总结项目进度和重要事项。
