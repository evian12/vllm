# CDB-AV 开发与版本规范

本仓库是 CDB-AV 的 vLLM 源码仓库。`~/CDB-AV`、`cdb-av-512-test` 的 tarball 解压目录和 `cdb_av_stepper_patch` 不是本仓库；不得把这些目录的代码或结果当成当前 Git 提交。正式实验优先从明确的 Git worktree 启动，并记录实际 Python 导入路径。若使用解压目录，必须另外保存包的提交号、所有本地差异与文件哈希，不能将一个提交号冒充完整运行源码。

## 开发前

1. 运行 `scripts/git_preflight.sh`，检查分支、HEAD、工作区、`myfork` 远端及同名远端分支。确认当前代码版本、上游更新和其他人的未提交修改。
2. 阅读本次相关的源码、调用链、已有测试和实验记录。确认服务正在使用哪个源码目录、GPU、端口与 Attention backend。
3. 有未提交修改时先明确所有权，保留原文件和结果；需要隔离时新建 worktree。不得使用 `reset`、`clean`、覆盖式 `checkout`、rebase 或 force push 处理他人的修改。

## 修改、检查与提交

1. 先说明目标、涉及文件、最小实现与正确性风险，然后修改。
2. 运行 `scripts/validate_changes.sh`。它检查 diff、Python/Bash 语法、版本工具单元测试和暂存内容；修改 `vllm/v1/` 时必须通过 `--pytest <相关测试路径>` 指定功能测试。必要时另做 GPU 端到端验证。不得用语法通过替代功能测试，也不得用 HTTP 200 替代有效 completion。
3. 查看完整 `git diff` 和 `git diff --cached`，确认没有无关修改、模型、数据集、大日志或秘密。每个提交只包含一个独立修改，用明确的 `feat:`、`fix:`、`test:`、`perf:` 或 `docs:` 信息。
4. 使用 `scripts/git_checkpoint.sh --message '...' --path <文件>` 仅暂存列出的文件；该脚本重复验证、正常提交、非强制推送到 `myfork` 的同名分支，并用 GitHub 返回的 SHA 核对。提交/推送失败时保留本地状态并明确报告。
5. 推送前读远端同名分支；有并行更新时先比较并保留双方修改。不能强推。通过检查的独立成果及时推送；未完成工作若必须保存，明确标记 WIP，放在独立开发分支。

## 分支与发布

- `main` 只接收完整验收的稳定版本；不得直接运行自动 checkpoint 到 `main`。
- `cdb-av/dev` 是日常集成分支。功能分支从其当前提交或明确记录的研究基线创建，完成相关测试后用正常快进/审查方式合入。
- `codex/refill`、`codex/dynamic-k`、`codex/kv-reuse` 是对应领域的建议分支名；有实际独立工作时再创建，不制造没有内容的分支。
- 老实验分支、快照、原始服务和历史结果均保留。不得为了同步而合并不同源码树。

## 实验追溯

正式实验前运行 `scripts/experiment_manifest.sh`，填入实际模型、GPU、H、K 策略、请求数、数据集、命令与结果目录。它从 Git 读取 SHA、分支和安全环境变量，默认拒绝脏工作区，且不会覆盖既有 manifest。原始日志留在服务器；将必要配置、摘要、失败数与复现步骤提交到 Git。请求级日志与客户端响应必须分开，记录 `requested_K`、Worker 的 `K_actual`、实际 Draft 步数及停止原因。客户端标签或 completion 长度不能代替 Worker 证据。客户端并发度、Target `max_num_seqs`、Draft 活跃行数与 Verify Batch 大小分别报告。

性能报告必须给出请求集、模型、backend、服务参数、源码 SHA、warmup、失败数、总输出 token、每轮 TPS/延迟和重复实验波动。只有相同负载和配置的配对实验才能支持性能归因。GPU 压测不是每次提交的必跑项；关键调度/KV 修改必须在正式性能实验前通过相应 GPU 正确性门槛。

## CDB-AV 专项验收

- **Dynamic K：** 当前 H=8 的实验允许服务端 cap=8，不设置客户端 1/2/3/4 的人为上限；证明请求级 `K_actual` 与低置信度等停止原因来自 Worker 实际 Forward。
- **Varlen：** 同一 Draft Batch 可有不同 K，完成请求退出后续有效 Forward，不能只改 mask/日志；输入、位置、注意力元数据、KV slot 和请求行映射正确，避免不必要 padding。
- **Persistent Refill：** 草稿槽位释放后合法等待请求实际进入后续 GPU Forward；新请求的 Target 首次 Prefill、Draft 上下文和 KV 所有权正确；新请求 Prefill 能与旧请求 Decode 合批。Ready Queue 满时未验证请求继续安全等待，不丢失、不重复验证、不提前提交 Draft token。取消、抢占、异常和结束时释放或失效 KV。
- **Target Verification：** 不同请求的候选长度与正式上下文对应；AR 与推测验证路径均正确处理接受、拒绝、bonus/correction 和状态迁移。未验证候选不能成为正式前缀。

报告必须分别标识：**代码已实现、单元测试通过、GPU 功能验收通过、GPU 性能测试通过**。四种状态不能互相替代。数值不一致应保留证据并分类，不能仅凭输出文本相似或逐字节差异推断正确性。
