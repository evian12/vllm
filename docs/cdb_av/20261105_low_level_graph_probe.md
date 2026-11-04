# CDB-AV 底层时间线与 Target 图路径试验（2026-11-05）

本轮保持 HF confidence 阈值 0.8、H=8、KV reuse 开启、单 GPU 顺序执行；没有调整请求选择、动态 H、补位或 KV 逻辑。原 18082 服务未被终止或重启。隔离测试使用物理 GPU 1、端口 18084，测试进程结束后已退出。

## 版本与结果位置

- 集成分支 `codex/full-system-audit-20261104`：后端启动参数修复为 `da2993c3fda1f41097050e027bfabbfe2932d0d2`，已推送 `myfork`。
- 独立试验分支 `codex/cdb-target-graph-experiment-20261105`：默认关闭的 Target 图路径门控为 `dc7b61ae2c2796b8212ae0a08bfc097346b8bfab`，已推送 `myfork`，**未合入集成分支**。配对实验运行时的源码为父提交 `cc1b57b718` 加上该提交所含的唯一源码差异；每个实验 manifest 保存了差异。
- 原始日志、客户端响应、指标与命令：`/home/yypan/CDB-AV/results/full_system_audit_20261104_1718/` 下的 `gpu_timeline_hf08_20261105_c/`、`gpu_timeline_native_k1_20261105/`、`gpu_timeline_steady_20261105_cdb/`、`target_graph_flashattn_smoke_20261105_b/`、`target_graph_pair_20261105_eager/`、`target_graph_pair_20261105_graph/`。每个目录的 `manifest.json` 记录完整启动命令和环境。

## CUDA 时间线的直接证据

在 CDB 的 **32 请求 × 64 输出 token** 短诊断中，用 Nsight Systems 只采集第 120–200 次 Scheduler 调用，时间线跨度 1.169 s。全部 GPU 活动的区间并集为 0.358 s，即采集区间内约 30.6%；这不是整轮服务 GPU 利用率。该采样窗口中：

| NVTX 区间 | 次数 | 主机墙钟合计 | GPU kernel 合计 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Target execute | 80 | 270.05 ms | 180.41 ms | 70 次没有任何 kernel；这 70 次墙钟合计仅 **3.19 ms** |
| Draft Forward | 37 | 731.08 ms | 174.09 ms | 每次 341 或 425 个 kernel；主机提交/等待占显著比例 |

因此，“跳过没有 GPU Target Forward 的执行入口”在该样本中最多能省下约 3.19 ms，无法解释主要吞吐差距。Draft Forward 的墙钟与 kernel 时间差很大，值得继续定位 Python/launch/同步与图捕获机会；Nsight 的追踪本身会增加开销，不能把差值全部归因于生产路径 CPU 空转。

同样的短采样中，原生 Spec K=1 的 80 次 Draft propose 墙钟合计 1,519.64 ms、GPU kernel 合计 395.48 ms，说明 eager Draft 启动开销并非 CDB 独有。但原生与 CDB 在此区间的实际 Target/Draft 调用结构不同，**不能把这两段短采样的吞吐直接比较**。一次 128 请求稳态 Nsight 采集虽到达预定第 1,500–1,580 次调度区间，工具仅输出 `.qdstrm` 原始流，未生成可解析报告；因此本轮没有稳态 GPU 时间占比结论。

## Target 图路径的功能与配对性能

源码原先要求 Target `model_config.enforce_eager`。隔离分支增加 `CDB_AV_TARGET_GRAPH=1` 门控：只有 Draft 的 speculative config 明确 `enforce_eager=true` 时才允许 Target 不强制 eager。原有开关关闭路径不变。初次启动时，**新 worktree 缺少 Git 忽略的本地 FLASH_ATTN 扩展**，自动回退 FLASHINFER；KV reuse 的布局断言正确阻止了执行。将现有同版本扩展以软连接接入试验 worktree，并使用 `--attention-backend FLASH_ATTN` 后，16/16 功能请求、1024 输出 token 完成，Target 图捕获完成，日志无 KV 布局、碰撞或运行异常。该功能 smoke 不等于严格输出等价或全生命周期 KV 验收。

同一源码、模型、GPU、backend、阈值、H、KV reuse、`max_num_seqs=4`、16 客户端并发、16 请求预热后，每种模式各运行两轮 **128 请求 × 128 输出 token**。每轮均 128/128 成功、16,384 输出 token、0 失败：

| 模式 | 每轮 output tok/s | 均值 ± 样本标准差 | 每轮 P95 延迟 |
| --- | --- | ---: | --- |
| Target eager + Draft eager | 162.976, 155.000 | 158.988 ± 5.640 | 15.141, 16.434 s |
| Target 图路径 + Draft eager dispatch | 159.419, 161.215 | 160.317 ± 1.270 | 16.035, 15.713 s |

均值差 **+0.84%**，低于本次重复波动；没有证据表明该图配置带来稳定吞吐或 P95 收益。尽管 Draft dispatcher 保持 eager，vLLM 的全局编译配置也对 Draft 模型执行 `torch.compile`，所以这不是只改变 Target CUDA Graph 的纯消融。接受 token / Draft token 比例：eager 两轮 70.51%、70.97%；图模式两轮 71.44%、70.81%。客户端工具只记录 token 数，不保存逐请求生成 token 序列，因此这里不能声称严格输出等价。试验代码仅保留在独立分支，暂不集成，也不开展 512 请求图模式压力测试。

现有正式 512 请求结果仍是 CDB 161.436 ± 1.667 tok/s（3 轮）；原生 AR 174.663、原生 Spec K=1 186.995 tok/s 各只有一轮同负载历史结果。CDB 仍未超过两者。

## 实验入口修正与后续

这份 vLLM 实际使用 `--attention-backend` CLI/config 选择后端，原 fair runner 中的 `VLLM_ATTENTION_BACKEND=FLASH_ATTN` 环境变量不是强制配置。集成分支现已在三种模式统一添加 `--attention-backend FLASH_ATTN`，两项离线 runner 测试通过；这只影响**未来新启动**的服务。

下一步应在无 Nsight 干扰下进一步拆分 Draft `_forward` 的模型调用、logits/置信度、metadata 构造与 CPU↔GPU 同步成本；如试验 Draft 图化，必须确保动态成员/异长输入、注意力元数据、KV 槽位和请求映射正确，并以独立可关闭分支先做 GPU 正确性验收。当前证据不支持只优化空 Target 入口、也不支持把 Target 图路径作为已成功的吞吐优化。
