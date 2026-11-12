# Colab MCP + LookAtMyQuant pilot

此目录提供 W-ModelForge 当前接入方案的 G4 配方。源码固定在本项目提交 `4dd8ec065d9308c8fe8c515e4fb086e8b298ee3d`，使用 `methods/lookatmyquant/` 中已维护的代码；不再克隆旧上游后另打补丁。

## 本机连接

安装 OAuth 依赖。官方 Colab CLI 包只用于 OAuth 库及公开的 installed-app 客户端资源，**运行时创建、代码执行和会话管理全部走官方托管 MCP**。

```sh
python3 -m venv .venv
.venv/bin/pip install google-colab-cli
export COLAB_MCP_ACCOUNT='your-google-account@example.com'
export GOOGLE_CLOUD_QUOTA_PROJECT='your-enabled-cloud-project-id'
.venv/bin/python colab/colab_mcp.py login
.venv/bin/python colab/colab_mcp.py tools
.venv/bin/python colab/colab_mcp.py call get_subscription
.venv/bin/python colab/colab_mcp.py call list_runtime_specs
```

Google Cloud 项目需启用 Colab API；账号完成 Google 要求的验证和授权。OAuth 文件仅保存在本机 `~/.config/colab-mcp/<account>.json`，权限0600。仓库不含任何账号、rclone凭据或私有执行记录。

按官方工具返回的 schema 和 eligible 规格调用 `create_runtime`，选择96GB级 G4；再用 `create_session` 创建会话。详见 [官方 MCP 参考](https://developers.google.com/colab/api/reference/mcp)。用实际返回的资源名替换下文 `runtimes/YOUR_RUNTIME` 与 `sessions/YOUR_SESSION`。

## 上传与运行

通过官方 `get_runtime.connectionInfo` 的文件接口上传，而非在执行代码中嵌入大文件：

```sh
.venv/bin/python colab/colab_mcp.py upload runtimes/YOUR_RUNTIME colab/run_quantization.py /content/run_quantization.py
.venv/bin/python colab/colab_mcp.py upload runtimes/YOUR_RUNTIME /private/path/selected-rclone.conf /root/.config/rclone/rclone.conf
```

rclone配置只导出本次选用的一个remote。上传后通过MCP执行 `os.chmod('/root/.config/rclone/rclone.conf', 0o600)`。不要上传本机OAuth文件。

`execute_code` 接收 `session`、`executionId` 和 `code`。每次新任务使用新UUID，把完整请求保存到忽略目录 `colab/runs/`。在会话中执行：

```python
from pathlib import Path
import os, subprocess

os.chmod('/root/.config/rclone/rclone.conf', 0o600)
Path('/content/model-forge').mkdir(exist_ok=True)
with open('/content/model-forge/pilot.log', 'a', buffering=1) as log:
    subprocess.run([
        'python3', '-u', '/content/run_quantization.py',
        '--model', 'Qwen/Qwen3.5-9B',
        '--drive-remote', 'your-drive-remote',
        '--drive-directory', 'model-quantization',
        '--run-id', 'pilot-001', '--stage', 'all',
    ], stdout=log, stderr=subprocess.STDOUT, check=True)
```

选择3.8-27B时，将模型名改为 `Qwen/Qwen3.8-27B`。两模型均自动启用CPU卸载；9B保持FP32，27B使用BF16，默认生成batch分别为4与8。先串行完成一个模型及回传，再运行另一个。27B构建前要求至少130GB空闲磁盘；确认9B完成并通过rclone校验后，才清理本次运行的展开临时权重。

工作目录为 `/content/model-forge/<run-id>/<model-name>/`。每阶段完成标记、状态和结果保存在 `output/`；大体积展开权重在 `work/`。不同实验使用不同run-id，避免混用完成标记。`--stage` 可单独选择 setup/calibrate/allocate/build/pack/evaluate/sync。已完成阶段重试时会跳过计算并重新校验回传；也可直接选择sync。

## 长执行和回传

MCP前端可能约600秒断开HTTP连接，运行进程仍可能继续。先下载日志/状态判断，**仅对仍在执行的任务重发原session、executionId及完全相同的code**以重新附着。不能用新UUID重新启动同一任务；已结束且最终结果已消费的请求也不能当作仍运行重发。

```sh
.venv/bin/python colab/colab_mcp.py download runtimes/YOUR_RUNTIME /content/model-forge/pilot.log colab/runs/pilot.log
```

脚本每阶段执行 `rclone copy` 与 `rclone check --one-way`；复制结果到 `<remote>:<directory>/<model-name>-LAMQ2-<run-id>`。产物包含源码/模型版本、环境、配方、许可证、修改说明和质量结果。完成评估及回传后释放本次GPU；脚本不保留或使用云端OAuth凭据来管理运行时。

## 配方与边界

128道MATH训练题自生成解答，最多1024新token；12个512-token窗口测Fisher，32个1024-token窗口做量化校准。线性层目标为2.7 bpw，2/3/4bit行级分配采用Fisher加权 `4^-bits` 代理；嵌入和独立输出头用lm4。trellis L12、输入Hadamard、LDLQ、数字token权重和QEP α=0.5，使用上游缓存EMA码本。

WikiText-2和英/中/代码留出窗口评估PPL/KL，另用上游独立216题集评估原始与量化模型。没有蒸馏；1024-token上限不能与原帖2048-token数学成绩直接比较。当前只量化文本部分。`packed/artifact.json` 和 `weights.bin` 是上游自定义格式；展开的dense.pt不代表压缩大小，尚未声称packed推理与展开权重的logits一致。

静态检查已完成；新优化的整体提速、27B实际峰值与质量仍待完整GPU运行。原有上游测试随历史保留，本项目不新增或执行单元/集成测试。来源与许可详见 [NOTICE](../NOTICE.md)。
