# 复现与回退
实验源码629c220b364e6a2806292a01c31b49caaeb46d27，工作树/home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113。
后续文档提交不改变运行源码；SOURCE.json含实际导入路径、客户端与模块SHA。
H8/HF.8/Kcap8/maxseq4/BF16/FA2/no-async固定。

serve_post_0.sh和serve_post_1.sh保存完整实际启动命令，含全部环境变量、模型、端口、Serve参数。
每次先检查GPU3显存/计算进程和18084，不能结束他人服务。
直接serve不负责关闭服务；优先使用带finally清理的隔离runner。

## 完整实际命令 post=0
```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113
env CDB_AV_PINNED_INPUTS=1 CDB_AV_DRAFT_POST_GRAPH=0 CDB_AV_DRAFT_POST_VALIDATE=0 CDB_AV_DRAFT_POST_VALIDATE_SAMPLES=64 CDB_AV_CARRY_VALIDATE=0 CDB_AV_EXACT_QKNORM=1 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 完整实际命令 post=1
```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113
env CDB_AV_PINNED_INPUTS=1 CDB_AV_DRAFT_POST_GRAPH=1 CDB_AV_DRAFT_POST_VALIDATE=0 CDB_AV_DRAFT_POST_VALIDATE_SAMPLES=64 CDB_AV_CARRY_VALIDATE=0 CDB_AV_EXACT_QKNORM=1 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 新目录隔离复跑
包装器只通过CPU dry-run/拒绝已有目录/拒绝相对路径检查；底层harness已用于真实GPU。
不要重跑归档driver的固定旧输出目录。每次on0/1必须另取全新out。

```bash
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/draft_post_graph_20261113/reproduce_isolated.py --mode validate --on 1 --out /home/yypan/CDB-AV/results/new_post_validate --dry-run
# 确认资源后去掉dry-run，32请求重型审计
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/draft_post_graph_20261113/reproduce_isolated.py --mode validate --on 1 --out /home/yypan/CDB-AV/results/new_post_validate
# 一次128预热+512正式
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/draft_post_graph_20261113/reproduce_isolated.py --mode perf --on 1 --out /home/yypan/CDB-AV/results/new_post_perf
```

## CPU/CUDA专项
CUDA命令仅GPU3空闲时运行；结束核对显存。
```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113
env CUDA_VISIBLE_DEVICES= CDB_AV_DRAFT_POST_GRAPH=0 CDB_AV_PINNED_INPUTS=0 CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av*.py
env CUDA_VISIBLE_DEVICES=3 CDB_POST_GPU_TEST=1 CDB_AV_DRAFT_POST_GRAPH=0 CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av_post_graph.py -k cuda
```

## 回退与计时口径
只设置CDB_AV_DRAFT_POST_GRAPH=0，保留Pinned1/QKboth1及既有Graph/H/K。
不reset、不覆盖checkout、不改依赖；旧2c12933工作树和备份保留。
CDB_AV_DRAFT_POST_VALIDATE=1只用于有界逐字节检查，默认64/max256，性能阶段0。
新Graph把head+confidence计入既有profile的logits区间；confidence_gpu只剩缓存取用，
不能把旧ON/OFF分段标签直接作为算子成本对照。正式TPS关闭重型审计。
输出为借用缓冲，下一次调用前消费；Worker单线程/单stream约束不可移除。

## 论文边界
AR/nativeSD/CDB都显式--no-async-scheduling，但原生CUDA Graph应独立保留并记录合适配置。
本轮只比较同一CDB代码的新开关，未重新运行原生AR/SD，不用历史异配置数字宣称优于它们。
