# CDB-AV 底层优化实测交付 — 2026-11-12

## 结论

沿此前约222 output tokens/s的底层版本继续，没有整合或修改网页版算法分支。
固定H=8、HF confidence=0.8、服务端K cap=8。
同一性能源码 `8df60e17ec889dd719973eb2adb57c15cd50fbfb` 下，
Draft Graph query bound4 / mixed keys12 为 **222.504±1.760 output tokens/s**；
query bound8 / mixed keys24 为 **233.425±1.834**，提升 **4.9084%**。
各轮请求P95均值11.2018s→10.6319s（约−5.09%）。±为三轮样本标准差。
这是当前已测最佳配置，不是全局最优证明，也不是与原生AR/Spec的比较。

推荐 `CDB_AV_DRAFT_RAGGED_MAX_QUERY=8`、`CDB_AV_RAGGED_GRAPH_MAX_KEYS=24`、
`CDB_AV_PACKED_METADATA=0`；其余使用附带真实启动命令。
Q是一次Forward输入的query长度上限，不是请求K；mixed keys24不含独立pure-decode图。
没有增加假Token行填充Graph。

## 版本与隔离

- 开发基线：`faa26fabc57d602cef4f55a797c39d10ba15e64b`，此前底层Graph/AR bypass/KV reuse/诊断版本。
- Query8：`783e0e8d557e763c66feabd493568dbdcd0c68e8`。
- Metadata与主性能实验：`8df60e17ec889dd719973eb2adb57c15cd50fbfb`。
- CPU shadow设备修复：`52c76cc22ca5f647fdc1a44a435b3d5139ddc5db`。
  后者通过完整CPU、GPU primitive与4请求模型测试，**没有重跑512性能**，不把8df性能冒充52c实测。
- GitHub：`codex/cdb-draft-q8-20261112`、`codex/cdb-packed-metadata-20261112`；
  正常push、核对远端SHA，文档提交另见GIT_DELIVERY.json。
- 主目录 `/home/yypan/CDB-AV/vllm` 保持 `e6b5defc8a4b15bace432c9f7b8b8b52c0a17a8f`，
  原有9项未跟踪备份/zip保留。没有reset/clean/覆盖checkout/force push，没有处理网页版模拟器。
- 工作目录 `/home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112`。
  六个实现/测试文件共306行增加、27行删除；完整CODE.patch保留。
- 全部GPU实验仅使用本轮独立GPU3（内部cuda:0）/18084，未动其他服务。
  最终18084关闭、GPU3=15MiB、无GPU3计算进程，见FINAL_GPU.json。

## 修改范围

| 文件 | 变化 |
|---|---|
| vllm/v1/spec_decode/cdb_av_config.py | query bound允许2/4/8，默认2保持；FA2/AOT约束覆盖8 |
| vllm/v1/spec_decode/cdb_av_refill.py | mixed图动态query域扩至8；opt-in packed metadata；一次Worker来源记录 |
| vllm/v1/spec_decode/cdb_av_metadata.py | 新鲜CPU shadow四字段16B对齐，一次H2D后精确dtype/shape/stride视图；显式CPU设备 |
| tests/v1/spec_decode/test_cdb_av_config.py | Query8配置和非法边界 |
| tests/v1/spec_decode/test_cdb_av_draft_decode_graph.py | query5–8的key、动态标量域回归 |
| tests/v1/spec_decode/test_cdb_av_metadata.py | 形状/dtype/字节、错误输入、默认设备、CPU/CUDA回归 |

不修改请求选择、H、HF停止规则、KV reuse判定、提交/失效/释放/Lease生命周期；
不关闭槽位碰撞、位置、请求/进度断言。Metadata默认OFF，原路径保留。
CPU shadow不复用尚在异步拷贝中的缓冲；8df实际Worker默认CPU有效，
52c修复全局默认设备非CPU时的健壮性。

## 公平性能对照

每轮512请求、closed-loop最多16个HTTP在途、每请求最大128输出token，
实际65536输出token/轮；服务端max_num_seqs=4。不是匀速到达，也不是GPU batch16。
固定模型、16类prompt生成器、greedy、seed42、请求索引、maxlen1024、BF16/FA2/TP1、GPU3；
native enforce-eager配合manual CDB Graph，无async/prefix cache。
每配置新服务、4个顺序32token上限warmup、连续3轮。
性能阶段关闭重型Trace/Graph验证/KV重放，保留有界聚合日志；GPU采样1Hz不是kernel-active时间。

### 同一8df代码，Target keys24不变

| Draft Q / mixed keys / packed | 三轮output tok/s | 均值±样本SD | 各轮P95秒的均值±SD |
|---|---|---|---|
| 4 / 12 / OFF（反向基线） | 220.586, 222.882, 224.044 | 222.504 ± 1.760 | 11.2018 ± 0.0426 |
| 8 / 24 / OFF（推荐） | 231.400, 233.903, 234.973 | 233.425 ± 1.834 | 10.6319 ± 0.0504 |
| 8 / 24 / ON | 232.387, 234.630, 233.365 | 233.461 ± 1.125 | 10.6915 ± 0.0898 |

三组4608/4608成功、589824输出token，无失败。
Metadata ON相对OFF只有+0.0151%，不能认定收益；P95略高，推荐OFF。

### 前置独立对照：同一783代码

| Draft Q / mixed keys | 三轮output tok/s | 均值±样本SD | 各轮P95秒的均值±SD |
|---|---|---|---|
| 4 / 12 | 220.442, 223.792, 221.907 | 222.047 ± 1.679 | 11.1829 ± 0.0439 |
| 8 / 12 | 221.038, 222.909, 222.996 | 222.314 ± 1.106 | 11.2131 ± 0.0386 |
| 8 / 24 | 228.807, 232.532, 232.511 | 231.283 ± 2.145 | 10.8202 ± 0.1343 |

另4608/4608成功。合计**18轮、9216/9216成功、1179648输出token**。
FAIRNESS.json独立核对4组完整命令、源码SHA和文件哈希，环境只差声明变量。
线程完成顺序、内部UUID、批次形状不完全确定，seed不能固定全部因素。
分组依次测试而非完全随机交错，后补反向基线支持方向，顺序/热状态偏差仍未完全排除。

## 正确性验收与边界

- 最新完整CDB CPU：**376 passed, 5 GPU-only skipped**；版本工具12项回归通过。
- 最新GPU metadata：**5 passed, 21 deselected**，逐字段dtype/shape/stride/原始字节与legacy相同，
  包含全局默认cuda设备时CPU shadow仍在CPU。
- 模型功能32+128+32+4请求，共196/196完成。
- **3378** 个Draft Graph检查：28层KV数值精确、Token精确、logits容差，
  包含静态与实际动态Metadata；Draft的torch.equal不是原始字节比较。
- **2003** 个Target检查：36层KV原始字节精确、Token精确、logits容差。
- **1070** 组真实Target interleave后的Carry快照一致，
  匹配恢复后实际Decode的Token/Position/Progress，最终Lease/Ready/inflight=0。
  最新4请求没有Carry，不能算新增Carry覆盖。
- 128请求诊断实际覆盖max query5/6/7/8，不是仅有客户端标签。
- completion API的effective_K/actual_draft_steps仍null，不当执行证据；
  Worker日志/聚合显示真实K=1..8，以及low_confidence/max_k停止。
- **严格整段输出等价未证明**：Metadata OFF/ON三轮最终文本hash仅163、152、161/512相同；
  Q4/Q8也不全相同。每Forward检查不等于最终输出证明，未把所有不一致归因于浮点近并列，
  未放宽测试。整体系统仍为 **Partial Implementation**。
- 未宣称全部自然内存压力/Scheduler路径/模型和并行组合已验收。

## 瓶颈证据与负结果

1. **Graph覆盖要与缓存容量配合。高置信实测。**
   Q4→Q8但keys12不变只有222.047→222.314（+0.12%）；Q8 keys12→24达到231.283（+4.03%）。
   2893个真实Forward上，离线分派先逐条核对真实Capture/Replay，冻结轨迹后minhits8预测
   Q8/12 Replay2243、Q8/24为2400。该预测只是形状诊断，不能当GPU性能模拟。
   更大query域受缓存工作集/捕获成本约束，不能自动加速。

2. **Metadata组件加速没有变成系统加速。高置信实测。**
   5轮交错micro：decode4约238→73μs（3.27×），mixed4约234→92μs（2.53×）；
   512×3只有+0.015%。不能外推micro加速，保留OFF。

3. **当前GPU kernel成本集中在线性层。中高置信，限短采样。**
   Nsight16请求/64token，此前16请求warmup，143518个kernel累计3.5914s：
   GEMM/GEMV92.819%、Attention2.238%、普通RMSNorm0.987%、RoPE0.641%。
   两模型kernel累计时间不是端到端占比；node tracing有扰动，未证明显存带宽饱和，
   也不把所有未标注范围统称Draft。

4. **Host memcpy时间不是PCIe耗时。高置信。**
   同段真实DMA H2D1.3002ms、D2H0.2888ms，Host cudaMemcpyAsync API却累计1.759s，
   含等待/追踪开销，不能当传输时间或全部可消除CPU时间。
   UniProc逻辑Worker RPC是本进程run_method，不能假设网络RPC成本。

5. **简单换BLAS没有明显收益。真实GPU负结果。**
   UnquantizedLinearMethod→dispatch_unquantized_gemm→torch.nn.functional.linear，
   当前PyTorch实际preference=Cublas。Cublas/CublasLt对15个层矩阵+6个LM-head形状，
   旋转权重组避免L2常驻偏差，CUDA Graph微测；输出全部torch.equal数值精确（未做原始字节比较），耗时差约−0.20%..+0.15%。
   不改生产后端/依赖，不追加无必要512实验；早期Default枚举对Lt不是生产对照，
   仅保留探索记录。

6. **KV复用真实生效，但不是本轮增益归因。**
   推荐组最后累计前缀5429442 reused/186170 prefill tokens，约96.685%；
   含warmup且非完整最终工作量，不能据此宣称KV复用TPS收益。
   同前缀90526 accepted/128161 drafted，不替代逐请求接受审计。
   未重写已安全的Lease/reuse路径。

7. **更大Graph仍需收益证据。仅优先级分析。**
   >8输入183次分散122个形状组，91组仅1次；
   冻结轨迹Q8 mixedkeys24→32预测Replay不增，Q16/24只增10次。
   未实现/宣称Query16 GPU支持，未盲目增加捕获成本。
   Target最后聚合仍有fallback_budget5271/27648，应独立核对热形状、捕获内存和成本，
   不能把该比例直接换算TPS提升。

8. **融合算子有候选，但先守住数值语义。**
   现有fused_qk_norm_rope在RoPE前保留FP32中间值，原路径norm输出BF16后再RoPE，
   不能假定字节相同。未贸然替换。GPU份额小，CPU launch收益仍待独立量化。
   当前没有全局最优证明；下一轮量化LM-head/GEMM与未覆盖Target形状，再决定最小修改；
   动态H、最终请求选择和阈值算法研究继续后置。

## 重现、回滚与失败保留

REPRODUCE.md和各轮*_launch.sh提供完整实际启动/客户端/测试命令。
开关回退MAX_QUERY=4、RAGGED_GRAPH_MAX_KEYS=12；PACKED_METADATA=0（默认0）。
独立worktree保留精确旧版，不覆盖脏目录、不改Git历史。

原始日志、逐请求响应、manifest、实际导入路径、GPU采样、SHA、校验器、失败日志均保留。
micro首次profile.py遮蔽stdlib cProfile，仅重命名本轮结果脚本后重试；
GEMM一次探针print键名错误修正重试，失败不计通过实验。
主工程、原生服务和其他用户进程未改变，结果软链接在代码目录可见。

# 重现与测试命令

本文件不自动执行GPU实验。先确认GPU3无计算进程、显存<1000MiB、18084未占用；
否则停止，不杀其他服务。GPU仅实验期间占用。
以下是8df性能实验完整实际命令；当前同worktree为52c及文档提交，性能仍归8df。
精确旧版应另建worktree并准备既有原生extension链接，不能覆盖当前源码。

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
ss -ltnp 'sport = :18084'
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
env PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

保留服务前台终端，确认健康、导入路径、FA2/BF16后另一个终端运行：

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
/home/yypan/CDB-AV/.venv/bin/python - <<'PYW'
import urllib.request,json
assert urllib.request.urlopen('http://127.0.0.1:18084/health',timeout=5).status == 200
for i in range(4):
    body=dict(model='qwen3-8b',prompt=f'Warmup inference {i}. Explain attention and KV cache.',
              max_tokens=32,temperature=0,stream=False,request_id=f'p0_perf_r0_warmup_{i}',
              vllm_xargs=dict(cdb_draft_k=8,cdb_trace_key=f'warmup:{i}'))
    req=urllib.request.Request('http://127.0.0.1:18084/v1/completions',
          data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:
        result=json.load(response)
    assert result['choices'] and result['usage']['completion_tokens']>0
PYW
for rep in 0 1 2; do
  /home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/stress_waiting_512.py \
    --port 18084 --requests 512 --concurrency 16 --max-tokens 128 \
    --run-id "perf_r0_repeat$rep"
done
```

各配置独立重启本次服务、4个warmup后3轮；其余命令保持：
Q4仅MAX_QUERY=4、RAGGED_GRAPH_MAX_KEYS=12；Metadata ON仅CDB_AV_PACKED_METADATA=1。
结束前台服务Ctrl-C，核对本次PID、端口和GPU显存；不使用pkill/killall。
实际harness为独立process group，finally只关闭自身。

最新实际测试：
```bash
cd /home/yypan/CDB-AV/worktrees/cdb-packed-metadata-20261112
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av_*.py
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av_metadata.py -k gpu_
```

GPU命令同样需隔离空闲。4请求run_hostfix.py固定结果路径并拒绝覆盖，重跑必须改新目录。
旧结果重算（无GPU）：
```bash
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/draft_q8_20261112/analyze_all.py
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/packed_metadata_20261112/analyze_all.py
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/packed_metadata_20261112/audit.py /home/yypan/CDB-AV/results/packed_metadata_20261112/validate8_meta1_hostfix
```

HTTP用vllm_xargs.cdb_draft_k传cap，不用嵌套extra_body；实际K由Worker confidence停止决定。
非streaming客户端没有TTFT/ITL，不从总延迟推算。
