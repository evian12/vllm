# CDB-AV 底层优化后续验证 — 2026-11-12

## 结论

继续此前底层分支，未整合网页版算法工作。本轮没有确认高于约233 output tokens/s的配置。
推荐仍为Draft query bound8 / mixed keys24、Target keys24、packed metadata OFF、H8/HF0.8/K cap8。
只落实一个不改变推理的CPU诊断热路径修补，不将其宣称为TPS提升。
整体仍为Partial Implementation；当前不能声称所有底层优化已完成或达到全局最优。

## Target Graph缓存容量公平对照

同一源码 **b4b6e8b0354b369f12170c17db352919603490fd**（包含52c CPU设备修补及文档），
Draft Q8/mixed24、metadataOFF，Target keys32先跑三轮，再keys24三轮。
同模型Qwen3-8B/Qwen3-1.7B、GPU3、FA2/BF16/TP1、H8/HF0.8/Kcap8，
max_num_seqs4、maxlen1024、eager外层/manual Graph、无async/prefix caching、seed42。
每服务4个短warmup后，512请求/C16闭环/每请求最大128，实际65536输出token/轮。
逐步Trace、验证、Profiler关闭；两边同样NVML1Hz采样。非streaming延迟，不包含TTFT/ITL测量。

| Target keys | 三轮output tok/s | 均值±样本标准差 | 各轮P95秒的均值±样本SD |
|---|---|---|---|
| 24 | 230.519, 233.820, 234.943 | 233.094 ± 2.300 | 10.6987 ± 0.1155 |
| 32 | 231.371, 234.000, 234.301 | 233.224 ± 1.612 | 10.8002 ± 0.1614 |

均值 **+0.0558%**，小于波动，32的P95反而略高，不采用32作为推荐配置。
共3072/3072成功、393216输出token。FAIRNESS.json核验源码/哈希/完整命令相同，
唯一环境差异CDB_AV_TARGET_GRAPH_MAX_KEYS=24/32。
分组顺序和线程批次形状/内部UUID无法完全固定，seed不能消除这些因素。
不能把单轮234.943或234.301当作稳定吞吐，也不与AR/原生Spec混比。

上一轮8df推荐配置233.425±1.834，本轮b4相同推荐配置233.094±2.300，结果接近；
本轮补充了52c代码所在提交的512重复测试。最新d7修补仅做下面的CPU/GPU功能测试，
**没有把b4性能冒充d7实测**。

## Target32实际GPU功能

- 128/128请求、16384输出token，真实捕获32个Target图。
- 新增至第25–32个的8组形状实际进行了417次独立对照。
- 1849次36层Target当前写入KV字节/Token/logits检查，
  2472次28层Draft数值/Token/logits及动态Metadata检查通过。
- 669次真实Target interleave后的Carry快照及恢复Decode关联通过。
- 最终Lease/Ready/inflight=0，未出现槽位冲突、位置或Metadata异常。
- Draft torch.equal为数值比较，Target KV检查为原始字节；不混称。
- 对实际轨迹逐条重放Target admission匹配Capture/Replay，额外形状数量改善不是TPS收益证明。
- 完整输出跨keys24/32仅158、156、159/512的文本hash相同，严格整段等价仍未验收，
  未把全部差异归因为浮点近并列。

## 最小代码修补：停止无效诊断格式化

代码 **d7fe5de4812ab9f53c98853be4da20af7914395d**，
分支codex/cdb-runtime-log-cap-20261112，正常push并核对远端。

- cdb_av_refill.py：_log_draft_graph_runtime_once达到32条配额后，立即返回，
  避免每次Forward先json.dumps(Graph key)和构造日志身份。
- test_cdb_av_draft_decode_graph.py：覆盖满配额后重复/新增key均不序列化，
  原有记录、32条上限和seen集合保持不变。
- 没有增减Forward、改变Tensor、K、H、采样、位置、KV或Scheduler行为；没有关闭安全断言。
- 50项相关测试+12项版本工具通过；完整CDB CPU **377 passed,5 GPU-only skipped**。
- 真实GPU16/16请求、1024输出token；
  恰好32条runtime记录，之后实际Draft Forward继续，总计157条Forward记录。
- 157次Draft与101次Target Graph检查通过，44次真实跨Target Carry关联通过，最终资源计数0。
- 这是确定消除冗余CPU工作的修补；没有为其声称可测TPS增益。

## 更正GEMM几何，并复查后端

此前独立探针把Target MLP当作intermediate14336，实际为 **12288**。
此前target_gate_up(28672,4096)、target_down(4096,14336)的6个M=1/4/16案例
只是代理形状，**不能作为实际Target MLP证据**。旧日志和原始数据保留。
真实512服务一直加载正确模型，Nsight真实模型测量和吞吐结果不受影响。

新探针从两个模型config自动推导，交叉检查16个safetensor权重头，覆盖10种实际矩阵：
Target gate/up(24576,4096)、down(4096,12288)；
Draft gate/up(12288,2048)、down(2048,6144)，另含QKV、O和LM head。
Draft词嵌入/LM-head共享由tie_word_embeddings和权重索引核实。
所有矩阵都在M=1/4/16微测，30案例均finite且输出**原始字节一致**。
使用随机BF16矩阵、旋转权重bank≥256MiB、CUDA Graph计时；不是实际模型输出或推理TPS。

29个案例变化在约±0.16%内，唯一draft_qkv/M1初测+5.77%有明显预热漂移：
Cublas21.42→20.62→19.51μs，Lt稳定19.50μs。
持续预热0.8s、10组交替顺序复查后：
Cublas中位 **18.92017μs**，Lt **18.93440μs**，
变化 **-0.0752%**，输出字节相同。
不把未复现的孤例推广到服务，不改变BLAS/依赖/生产配置。

## QK Norm + RoPE原生融合组件探索

仅在独立随机QKV张量上调用已有CUDA算子，未替换Draft或Target模型路径。
2个模型头配置×6个行数(1/4/8/16/64/256)×2类权重(ones/实际layer0 norm)共24案例；
BF16，位置0..1023，双方含相同输入reset copy，
每图32次组合、20次Replay/样本、5样本，减少Host replay间隙影响。
这是组件区间，不是系统吞吐。

学习权重/4行：
- Draft：5.5259→2.5840μs；
  Q rel_l2=0.00356504，K=0.00370594。
- Target：5.6989→2.4400μs；
  Q rel_l2=0.00348831，K=0.00275965。
- 整体约1.99–2.37倍组件加速；24例Q/K均非字节相同，V切片均字节不变。
  Draft最大绝对Q误差0.125/K误差1.0，Target最大Q0.0625/K0.5；
  绝对值受学习权重尺度影响，需连同RMS/相对L2阅读原始RESULTS.json。
- 源码显示native RMSNorm在乘权重前后有BF16中间舍入；fused实现保留FP32，
  且warp32归约与原CUB块归约不同。不能只加一次末端cast就认定数值等价。
- V不变不代表KV Cache生命周期验证；这是中间张量试验。
  没有用微测放宽现有Graph/KV断言，也没有将其当成正式模型性能收益。

## 当前瓶颈判断及下一步

已证实：Draft更大query图域配合混合缓存容量有约4–5%收益；
Metadata合包和Target24→32没有确认端到端收益；BLAS简单切换未复现收益。
此前真实Nsight的GEMM/GEMV占两模型kernel累计时间92.8%，不是端到端比例，也未证明显存带宽饱和。
CPU占用可包含同步等待，不能仅凭100% CPU或80% NVML利用率定位CPU瓶颈。

仍开放的底层候选：**保持原BF16逐次舍入和归约语义的QK Norm/RoPE融合实现**。
先独立验证Tensor布局、V不变、极值/零/边界输入及数值范围，再判断是否可接入模型；
之后才做实际GPU模型正确性与重复TPS/P95对照。当前没有实现该新kernel。
原生融合组件较快不代表整体值得启用，须考虑安全门槛及GPU/CPU关键路径；
固定H，最终请求选择与算法调整不在本轮范围。

22:00已设单次收敛复查。根据有证据的候选清单决定继续或收敛，
不因为六轮都成功就宣称所有优化完成。

## 资源、清理、重现

GPU3已回15MiB，18084关闭，无本轮GPU计算进程。所有自建服务/探针退出，
原服务和他人进程未结束。一个远端已结束但本地仍残留的本轮SSH客户端已单独关闭；
六轮通过由完整原始结果/来源/清理文件验证，不以该残留通道退出码代替实验状态。

不删除历史结果、备份、未提交修改和不确定所有权内容。本轮只收尾自建进程；
源代码和结果均保留并归档，不贸然删除研究分支。
完整实际启动命令见REPRODUCE.md及每轮*_launch.sh；
回退Target keys24，metadata0。CPU日志修补可通过独立b4 worktree回到旧实现，
不reset现有目录。性能数据SHA=b4，最新功能代码SHA=d7，交付文档提交另记。

# 本轮实际启动和测试命令

以下用于复现，当前不会自动启动测试。先确认GPU3空闲/无计算进程和18084关闭；只关闭自己本轮进程。
所有full launch均由实际harness输出保存，性能源码b4、功能修补d7；当前HEAD可能包含后续文档提交。

## 性能Target keys24（GPU3/18084）

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
env PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 性能Target keys32（GPU3/18084）

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
env PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=32 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 日志修补GPU验证实际启动命令

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-runtime-log-cap-20261112
env PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-runtime-log-cap-20261112 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=1 CDB_AV_TRACE_TARGET_GRAPH=1 CDB_AV_TARGET_GRAPH_MAX_KEYS=32 CDB_AV_TARGET_GRAPH_MIN_HITS=2 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=1 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=1 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=1 CDB_AV_DRAFT_GRAPH_VALIDATE=1 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=2 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 客户端、预热和测试

各性能服务4个顺序相同warmup后3轮。warmup HTTP JSON见harness.py（与上一轮一致）。
性能客户端：
```bash
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
for rep in 0 1 2; do
 /home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/stress_waiting_512.py --port 18084 --requests 512 --concurrency 16 --max-tokens 128 --run-id "perf_r0_repeat$rep"
done
```

完整客户端JSON实际由vllm_xargs传K cap8，Worker HF0.8决定实际停止。
GPU验证16请求64输出，实际命令在runtime_log_cap/harness.py生成日志中保留；
run_gpu.py固定输出名、拒绝覆盖，重跑须使用新的输出目录。

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-runtime-log-cap-20261112
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av_*.py
```

重算现有数据（无GPU）：
```bash
python3 /home/yypan/CDB-AV/results/target_cache_20261112/analyze.py
python3 /home/yypan/CDB-AV/results/target_cache_20261112/dispatch_replay.py
```

独立算子探针的完整命令/来源见各probe.py和controller；仅空闲GPU3运行，不与服务并行。
结束在服务前台Ctrl-C，核对本次PID/端口/显存。历史原始结果不删除。
