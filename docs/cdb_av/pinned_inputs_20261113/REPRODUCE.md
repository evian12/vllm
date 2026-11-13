# 启动、测试、复跑、回退

运行源码1096679a1401517852a18248564bc8f66b181508，后续文档提交不改变推理代码。
下列在服务器执行，仅GPU3/18084；先检查GPU进程/显存/端口。不能停止其他服务。
直接serve命令不负责结束服务；推荐优先使用带finally清理及输出目录保护的隔离runner。

## 完整开启命令

```bash
nvidia-smi --id=3
ss -ltnp 'sport = :18084'
cd /home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113
env CDB_AV_PINNED_INPUTS=1 CDB_AV_EXACT_QKNORM=1 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 完整关闭/回退命令

先正常结束自己启动的服务并确认资源空闲；只改变新优化开关0，保留旧QK融合、H8/HF0.8及原Graph设置。
不reset、checkout、clean或改依赖；原f3e工作树完整保留。

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113
env CDB_AV_PINNED_INPUTS=0 CDB_AV_EXACT_QKNORM=1 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 安全复跑（新结果目录）

包装器只做过CPU dry-run/已有目录拒绝/相对路径拒绝检查；
其底层harness已用于本轮真实GPU，不能把包装器dry-run当GPU验收。
不要直接重跑原有固定输出driver，以免覆盖原始日志或被existing-dir检查拒绝。

```bash
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/pinned_inputs_20261113/reproduce_isolated.py --mode validate --pin 1 --out /home/yypan/CDB-AV/results/pinned_new_validate --dry-run
# 确认新目录与资源后，去掉dry-run，运行32请求真实输入/Graph审计
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/pinned_inputs_20261113/reproduce_isolated.py --mode validate --pin 1 --out /home/yypan/CDB-AV/results/pinned_new_validate
# 一次新服务128warmup+512正式；重复必须另取不同out，pin0/1控制开关
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/pinned_inputs_20261113/reproduce_isolated.py --mode perf --pin 1 --out /home/yypan/CDB-AV/results/pinned_new_perf
```

每次保存Source/ClientSHA、导入路径、实际完整launch/config、manifest、请求结果与CLEANUP。
CPU/GPU前置检查失败即退出，不换卡/杀其他进程。GPU服务仅属于该runner，finally关闭后检查端口与显存。

## CPU测试

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113
env CUDA_VISIBLE_DEVICES= CDB_AV_PINNED_INPUTS=0 CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av*.py
env CUDA_VISIBLE_DEVICES= CDB_AV_PINNED_INPUTS=0 CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD PATH=/home/yypan/CDB-AV/.venv/bin:$PATH bash scripts/validate_changes.sh --pytest tests/v1/spec_decode/test_cdb_av_inputs.py --pytest tests/v1/spec_decode/test_cdb_av_draft_decode_graph.py
```

## GPU传输原语（无模型HTTP服务）

仅GPU3无计算进程且端口18084空闲时执行，保存到全新的日志路径。

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113
env CUDA_VISIBLE_DEVICES=3 CDB_PINNED_GPU_TEST=1 CDB_AV_PINNED_INPUTS=0 CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av_inputs.py -k 'cuda_fields or pending_source or cpu_shadow_stays or bounded_gpu_validation'
```

## 诊断/论文设置

- `CDB_AV_VALIDATE_METADATA=1`启用新传输前64次真实字段字节检查（另有旧Metadata检查），
  可用`CDB_AV_INPUT_VALIDATE_SAMPLES=1..256`调整有界样本数。性能阶段0。
- 物理Carry审计另开`CDB_AV_CARRY_VALIDATE=1`；Target有界计时
  `CDB_AV_TARGET_GRAPH_TIMING=1`必须同时`CDB_AV_TARGET_GRAPH_VALIDATE=1`。
  本轮16请求/64输出、8个Target replay样本，全部完整命令见carry_validate_pin1/validate_r0_launch.sh。
  这些重型诊断不能当生产TPS对照。
- 本论文AR/原生SD/CDB显式统一`--no-async-scheduling`；模型GPU Forward串行。
  新输入DMA的non_blocking不是原生async scheduling，二者不混称。
- 私有QK库仍指向此前b148binary，source receipt保持匹配；本轮没有构建新CUDA.so。
