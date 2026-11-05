# CDB-AV 性能复审（2026-11-05）

## 结论

当前瓶颈不是 Scheduler 本身的 CPU 时间，也不是 Draft 模型的 GPU 计算量接近 Target。主要问题是：小批量 eager 模式将 Draft 的轻量 GPU 计算放大成昂贵的逐步前向；CDB 的固定 H=8、Target-first 准备与 Ready/Carry 状态使大量 Target 行走 AR 而非批量 Verify，动态 K 带来的额外接受 token 很少，因而无法抵消 Draft 前向及系统管理成本。这个结论有稳态 CPU profile、阶段计数及同负载 Scheduler 行数支持；GPU kernel 与主机开销的精确稳态拆分仍需进一步测量。

## 版本与环境

- 审查源码：`/home/yypan/CDB-AV/worktrees/full-system-audit-20261104`，HEAD `91c8ea1e95832015a23563329c88fe479996a92b`，工作树干净，远端 `myfork/codex/full-system-audit-20261104` 同 commit。
- 新诊断：同一 HEAD、Qwen3-8B / Qwen3-1.7B、物理 GPU 1、FLASH_ATTN、eager、`max_num_seqs=4`、固定客户端并发 16、128 请求、每请求 128 输出、预热 16；CDB 为 H=8、HF threshold=0.8、KV reuse ON。原服务 GPU 3/18082 未修改，独立诊断服务 GPU 1/18084 已关闭。
- 512 性能比较：CDB commit `814a76469eaa1927092d3b3aa721e0cc66ccf363`，原生 K=1/4 与 AR commit `d378fbaccf50f7ba711d5307ffeb37be084cb33a`；均为同模型、同请求集和限制，服务日志确认 FLASH_ATTN。基准代码 commit 不同，原生 K=1 与 AR 仅一轮，不能当作统计显著性结论。

## 吞吐现状

| 配置 | 512 请求 output tok/s | 重复 | 备注 |
|---|---:|---:|---|
| CDB HF 0.8、H=8、KV reuse ON | 161.436 ± 1.667 | 3 | 每轮 512/512 成功，65,536 输出 token |
| 原生 Spec K=1 | 186.995 | 1 | 比 CDB 高 15.8%（以 CDB 为分母） |
| 原生 AR | 174.663 | 1 | 比 CDB 高 8.2% |
| 原生 Spec K=4 | 135.691 | 1 | CDB 更快 |
| 原生 Spec K=8 | 94.603 ± 0.212 | 3 | CDB 更快 |

原始结果：`/home/yypan/CDB-AV/results/full_system_audit_20261104_1718/{cdb_hf08_lowlog_512,fair_native_spec_k1,fair_native_spec_k4,fair_native_spec,fair_native_ar}`。新同代码 128 请求单轮计数探针得到 CDB 161.816、原生 K=1 193.112 tok/s；这是诊断运行，不能代替 512 的重复性能实验。

## 同代码、同负载的 Target 调度计数

探针每 32 次调度写一次快照；请求前后差值最多约 ±31 次调度和相应行数，不影响下列量级判断。`scheduled_rows` 是所有实际调度的 Target 请求行，包含 Prefill/AR/Verify；`verify_rows` 是携带 speculative candidates 的请求行。

| 指标，128 请求 / 16,384 输出 token | 原生 K=1 | CDB HF 0.8 |
|---|---:|---:|
| Scheduler/Target execute 调用 | 2,400 | 2,848 |
| 其中无 scheduled token | 1 | 346 |
| 实际 Target 前向机会 | 2,399 | 2,502 |
| Target 请求行 | 9,477 | 9,170 |
| 带验证候选的请求行 | 9,349（98.6%） | 4,247（46.3%） |
| 候选 token | 9,349 | 10,258 |
| 实际 Target 前向平均请求行 | 3.95 / 4 | 3.67 / 4 |

CDB 比原生多约 4.3% 的有效 Target 前向、9.7% 的候选 token，却只有不到一半 Target 行携带候选；其余约 4,923 行是 Prefill 或 AR（减去新请求首轮 Prefill 后，AR 仍占绝大多数）。原生 K=1 的非 Prefill 行几乎全部带候选。CDB 的更长 K 集中在较少 Verify 行上，没有明显减少 Target 前向总数。这是系统设计与计算效率层面的主要差异。

探针文件：`/home/yypan/CDB-AV/results/full_system_audit_20261104_1718/schedule_count_{native_k1,cdb_hf08}_20261105/`，含启动 manifest、预热前后计数、客户端结果、服务日志。该探针不改变 vLLM 源码。

## Draft 单步为何没有想象中便宜

当前 HEAD 的稳态 `cProfile` 覆盖第 1500–1700 次调度：CDB 200 次 Draft 前向累计 3.786 s（18.9 ms/次），180 次实际 Target 模型前向累计 4.230 s（23.5 ms/次）；原生 K=1 的 200 次 Draft proposal 累计约 3.69 s，200 次 Target 模型前向约 4.50 s。两种 Draft 的 CPU 可见路径成本接近，未发现 CDB 独有的单次 Draft 大幅退化。CDB 的 Target/Draft 模型路径占该 profile 约 81%；Scheduler `schedule()` 仅 0.074 s，Draft metadata 0.059 s。`token_fingerprint()` 0.586 s 累计、`ReuseDraftQueue.reconcile()` 0.525 s 累计，是可优化的次级成本；这两项与上层调用重叠，不能相加当作独立墙钟。

早先 32×64 的短 Nsight 样本中，Draft GPU kernels 约 4.7 ms/次，Target GPU kernels 约 18 ms/次；GPU 计算的确明显更轻。但那是过渡阶段短样本，不能推出稳态 GPU 利用率。Draft CPU 路径约 15–19 ms/次，提示 `--enforce-eager` 下 28 层模型的 kernel launch、Python 框架和逐步同步使其 GPU 优势难以转化为相应的墙钟优势。需要稳态短窗口 GPU trace 进一步拆分。

旧阶段探针（不同诊断插桩，**不能与新计数运行直接做 TPS 因果比较**）在同一 128×128 负载记录：3,033 次实际 Draft Forward、3,320 次 Draft 窗口步、约 287 次 Ready-only 空步；Draft 前向累计 45.82 s，Target execute 累计 51.22 s，Scheduler schedule 累计 0.698 s。Draft Forward 的有效请求行共 10,820，即 3.57/4，槽位利用率约 89%。连续补位已有实际工作量，不能把吞吐差距简单归结为 Draft Batch 空槽。混合 Prefill/Decode 约 2,202 次，平均墙钟约 14.8 ms/次；纯 Decode 632 次，约 16.1 ms/次。因此“每轮全量 Prefill”已被 KV reuse 改善，剩余混合 Prefill 不是首要单次耗时热点。

## 算法与状态机原因

- 当前 `finish_window()` 必须恰满 H=8 才发布 Ready；Window 内已 Ready 的请求仍按窗口边界等待。`PersistentDraftSession.step()` 在全 Ready 时发无 GPU 前向的 idle tick。它造成调度/RPC 与等待，但空步本身不是 Draft GPU 空转。
- Scheduler 跳过 Carry/Frozen，请求在 Target-first Prefill/首 token 后才有资格进 Draft；Running 先于 Waiting 调度。旧详细样本 3,324 个调度点的 Ready Queue 平均 4.19，平均选中 1.34，平均延后 2.85；这说明候选积压和验证批量不足，但没有当前版本的 Ready 等待时间直方图，不能直接量化延迟贡献。
- CDB 的 HF 0.8 在 512×3 累积中，实际 K=1 约 46.6%，平均 K 约 2.42；候选接受率约 70.6%，按 Prometheus 三轮增量，每轮平均约 41.2k 候选、29.1k 接受，约占 65.5k 输出的 44.4%。原生 K=1 的日志接受率约 75.5%，其接受 token 对总输出的贡献接近 CDB。更长 K 没有获得足够多的额外接受 token 来支付逐步 Draft 与更宽 Target 验证的成本。此处不是动态 K 算法研究结论，只说明当前系统负载下的投入/产出。
- KV reuse 的累计 `p2_reuse_plan` 记录约 5.27M 逻辑前缀 token 复用与 189.6k 增量 Prefill token（含三轮及预热）；不能把该比值当成 GPU 加速率。此前相同阈值 0.4 的 ON/OFF 三轮 TPS 为 138.204 vs 114.175，说明复用机制有实际收益，但不代表阈值 0.8 的增益精确相同。

## 优先排查/实验

1. **Draft 前向的主机/GPU 拆分。** 在同一 H=8、threshold=0.8、相同请求集下，用可解析的稳态短 Nsight/CUPTI 采样定量 kernel、memcpy、CPU launch、同步和 GPU idle。然后做 Draft-only graph/固定形状桶的最小可回滚原型；保持 ragged metadata、KV slot、token/K 结果一致。旧 Target graph 混合编译实验的 +0.84% 在噪声内，不能作为 Draft graph 无益的证据。
2. **AR-only Target 行归因。** 用低开销聚合计数把非验证行拆为 Target 首次 Prefill、正式 AR Decode、Ready 未发布、Ready 因容量/预算延期、Carry 冻结、没有 eligible Draft 等原因。目标是解释当前约半数非 Verify 行，并找出不改变正确性的批量 Verify/Target-first 补位改进。现有计数还不能证明具体哪一种机制占主导。
3. **每输出 token 的完整工作量。** 同口径计数 Draft GPU Forward、有效行、候选/接受分布、Target 实际 Forward、Ready 等待与 Target Batch，做一轮小规模因子隔离，再做 3–5 次 512 配对验证。保留固定 H=8、阈值 0.8；不先调动态 H 或最终请求选择算法。
4. **次级 CPU 路径。** 在确保请求前缀版本及失效安全的前提下，减少重复 `token_fingerprint`/context 重建与 Worker RPC 包装；以成对测试确认收益。不要先重构 Scheduler：其可见时间远低于模型路径。

本轮仅新增独立诊断脚本和结果，没有修改、提交或重启集成工作树/原服务。两个本次创建的 18084 服务均已退出；原 18082 服务仍在运行。
