# CDB-AV Git 与实验版本记录

源码仓库是 `~/CDB-AV/vllm`，GitHub fork 是 `git@github.com:evian12/vllm.git`（remote `myfork`）。上游 `origin` 指向 `vllm-project/vllm`，日常推送不要使用 `origin`。`~/CDB-AV/cdb-av-512-test` 是解压的测试源码，没有 `.git`；它的运行结果必须另记包提交号、隔离目录修改和实际导入路径。`~/CDB-AV/cdb_av_stepper_patch` 也是仓库外目录。

工作前在实际 Git worktree 中运行：

```bash
scripts/git_preflight.sh
git status --short
git diff --check
```

完成独立修改后运行快速检查。核心调度/KV 代码还需要相关功能测试，GPU 性能测量另行执行：

```bash
scripts/validate_changes.sh --pytest tests/v1/spec_decode/cdb_av
git diff
scripts/git_checkpoint.sh --message 'fix: describe one verified change' \
  --path vllm/v1/spec_decode/example.py \
  --path tests/v1/spec_decode/cdb_av/test_example.py \
  --pytest tests/v1/spec_decode/cdb_av
```

`git_checkpoint.sh` 只暂存列出的路径；运行 diff、语法、版本工具单元测试与指定 pytest；拒绝模型、日志、数据集、凭证及大文件；正常 commit 和 push；最后通过 `git ls-remote` 核对远端 SHA。远端已经前进时直接报错，人工分析后再合并，不自动覆盖。`main` 禁止直接 checkpoint。

正式实验建议从干净 worktree 启动，运行前生成 manifest，例如：

```bash
CUDA_VISIBLE_DEVICES=1 CDB_AV_WINDOW_H=8 CDB_AV_K_POLICY=hf_confidence \
scripts/experiment_manifest.sh \
  --experiment-name refill_h8_512 \
  --vllm-version 0.15.1 \
  --target-model /disk14T/yypan/models/Qwen3-8B \
  --draft-model /disk14T/yypan/models/Qwen3-1.7B \
  --gpu 'RTX 4090 physical GPU 1' \
  --window-h 8 --dynamic-k-policy hf_confidence \
  --request-count 512 \
  --dataset 'stress_waiting.py generated prompts (16 templates)' \
  --source-import-path /home/yypan/CDB-AV/vllm/vllm/__init__.py \
  --command 'bash scripts/cdb_av/run_512_stress.sh' \
  --test-status not_run \
  --run-dir /home/yypan/CDB-AV/results/example_run
```

以上是命令格式示例，不代表该配置已运行；实际值由执行者按服务、源码和请求集填写。manifest 默认拒绝未提交改动、拒绝覆盖已有文件，并保存 Git SHA、分支和安全环境变量。若必须研究脏工作区，可使用 `--allow-dirty`，同时记录 diff 和未跟踪文件哈希；结果不得被描述为某个干净提交的复现。

公开仓库中的分支可以通过 `https://github.com/evian12/vllm/tree/<branch>` 查看。Git 推送和 `git ls-remote` 成功只证明 GitHub 仓库同步；ChatGPT 是否能读取私有/受限代码取决于用户在 ChatGPT 中单独连接并授权 GitHub。当前 fork 为公开仓库，仍需在目标 ChatGPT 会话里确认它实际读取了所需的分支与提交，而不能只看 `main`。
