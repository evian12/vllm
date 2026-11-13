# 完整启动、复跑及回退

所有命令在rtx4090服务器执行。仅使用GPU3/18084；先确认无其他计算进程、显存与端口空闲。
不要终止任何不属于本轮的服务。以下直接serve命令不自动停止服务，建议优先使用自动finally清理的隔离复跑入口。

## 推荐加固版：完整性能配置启动

正式实测源码c7531aaef4438b138c600e22aae384519a4522f6；私有二进制
b148b8541950b326f49ea4222b136d336bac5fefeaeab012b14bf38606180f5b。
当前选定worktree可能在之后增加文档提交，核心文件SHA见SOURCE_SHA256.json/运行目录SOURCE.json。
实际启动导入记录见各IMPORTS.log，Worker二进制回执见server.log的CDB_EXACT_QK runtime。

```bash
nvidia-smi --id=3
ss -ltnp 'sport = :18084'
cd /home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113
env CDB_AV_EXACT_QKNORM=1 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

## 回退：同一源码关闭新融合，完整命令

先结束自己启动的服务并确认GPU3/18084空闲，不能盲目kill。
仅EXACT开关0，其他已选底层优化不变，不需要reset、checkout或改依赖。

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113
env CDB_AV_EXACT_QKNORM=0 CDB_AV_EXACT_QKNORM_MODE=both CDB_AV_EXACT_QK_LIB=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2/cdb_exact_qknorm_20261113.so CDB_AV_EXACT_QK_VALIDATE=0 CDB_AV_EXACT_QK_VALIDATE_SAMPLES=32 PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113 CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN CDB_AV_TARGET_GRAPH_DYNAMIC=1 CDB_AV_TARGET_GRAPH_TIMING=0 CDB_AV_PACKED_METADATA=0 CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 CDB_AV_OUTPUT_BUDGET_CAP=0 CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_VALIDATE=0 CDB_AV_TRACE_TARGET_GRAPH=0 CDB_AV_TARGET_GRAPH_MAX_KEYS=24 CDB_AV_TARGET_GRAPH_MIN_HITS=8 CDB_AV_DRAFT_RAGGED_MAX_QUERY=8 CDB_AV_REFILL=1 CDB_AV_PERSISTENT=1 CDB_AV_KV_REUSE=1 CDB_AV_INTRA_WINDOW=1 CDB_AV_LOW_LOG=1 CDB_AV_K_POLICY=hf_confidence CDB_AV_HF_THRESHOLD=0.8 CDB_AV_AR_TARGET_BYPASS=1 CDB_AV_WINDOW_H=8 CDB_AV_STEP_MODE=1 CDB_AV_VARLEN=1 CDB_AV_SPLIT_RPC=1 CDB_AV_WORK_QUEUE=1 CDB_AV_INITIAL_ADMIT=3 CDB_AV_ALLOW_FRESH_WAITING=0 CDB_P1_VIRTUAL_H=0 CDB_AV_TRACE_STEPS=0 CDB_AV_REFILL_TRACE=0 CDB_AV_REUSE_REPLAY=0 CDB_AV_REUSE_VALIDATE=0 CDB_AV_VALIDATE_METADATA=0 CDB_AV_DRAFT_DECODE_GRAPH=1 CDB_AV_TRACE_GRAPH=0 CDB_AV_DRAFT_GRAPH_VALIDATE=0 CDB_AV_DRAFT_RAGGED_GRAPH=1 CDB_AV_RAGGED_GRAPH_MIN_HITS=8 CDB_AV_RAGGED_GRAPH_MAX_KEYS=24 CDB_AV_DRAFT_RAGGED_DYNAMIC=1 /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

如需整个原始底层版本，41483eebf0deda8dfcfbbe7dfa06b7314492b1f3保存在Git历史及此前交付；
只能另建worktree，不能覆盖工作区。简单关闭开关已恢复本次改动前的native Norm/RoPE执行路径。

## 不覆盖结果的隔离复跑入口

脚本检查--out必须是不存在的绝对目录，复制已验收的driver到新目录；
GPU/端口预检失败就退出，不会换卡或杀其他进程。服务仅由该driver创建及finally结束。
下面都是命令示例，本轮没有自动追加这些新实验。

```bash
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/exact_qknorm_20261113/reproduce_isolated.py --mode validate --out /home/yypan/CDB-AV/results/qk_my_new_validate --dry-run
# 确认新目录与资源后，去掉--dry-run执行32请求重型功能审计
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/exact_qknorm_20261113/reproduce_isolated.py --mode validate --out /home/yypan/CDB-AV/results/qk_my_new_validate

# 一次新服务128warmup+512正式性能；不同重复使用不同--out
/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/exact_qknorm_20261113/reproduce_isolated.py --mode perf --out /home/yypan/CDB-AV/results/qk_my_new_perf
```

本轮实际driver：run_model.py、run_pair.py、run_pair_v2.py、run_guard_repeats.py。
它们原始固定输出目录已完成，不要直接重跑以免覆盖日志或被existing-dir保护拒绝；用上面的新目录入口。
REPRODUCER_TESTS.json覆盖dry-run无目录/无GPU、拒绝已有目录、拒绝相对目录。
新入口只做过CPU dry-run保护检查，核心隔离harness用于真实实验；不声称包装器另做过GPU验收。

## CPU/协议检查

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113
env CUDA_VISIBLE_DEVICES= CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q tests/v1/spec_decode/test_cdb_av*.py
env CUDA_VISIBLE_DEVICES= CDB_EXACT_BUILD_DIR=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q experiments/cdb_exact_qknorm/test_cpu.py
env CUDA_VISIBLE_DEVICES= CDB_AV_EXACT_QKNORM=0 PYTHONPATH=$PWD PATH=/home/yypan/CDB-AV/.venv/bin:$PATH bash scripts/validate_changes.sh --pytest tests/v1/spec_decode/test_cdb_av_exact_qknorm.py
```

## GPU组件复核

只在资源预检通过后执行，无HTTP/模型服务，进程退出即释放其CUDA上下文。

```bash
CDB_CHECK_OUT=$(mktemp -d /home/yypan/CDB-AV/results/exact_qk_component.XXXXXX)
cd /home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113
env CUDA_VISIBLE_DEVICES=3 CDB_EXACT_BUILD_DIR=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python experiments/cdb_exact_qknorm/validate_gpu.py --out "$CDB_CHECK_OUT/operator" > "$CDB_CHECK_OUT/operator.log" 2>&1
env CUDA_VISIBLE_DEVICES=3 CDB_EXACT_NEGATIVE_GPU=1 CDB_EXACT_BUILD_DIR=/home/yypan/CDB-AV/results/exact_qknorm_20261113/build_v2 PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python -m pytest -q experiments/cdb_exact_qknorm/test_gpu_api.py > "$CDB_CHECK_OUT/negative.log" 2>&1
nvidia-smi --id=3
```

原始负测试日志negative_gpu_api.log、独立gpu_v2/ACCEPTED.json、边界BOUNDARY_GPU.json。
无效GPU position的device assert保留；本轮没故意触发来污染CUDA上下文，合法0/1023边界已测。

## 从源码重新构建（独立目录，禁止覆盖旧二进制）

```bash
CDB_REBUILD_OUT=$(mktemp -d /home/yypan/CDB-AV/results/exact_qk_rebuild.XXXXXX)
cd /home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113
env CUDA_VISIBLE_DEVICES= CUDA_HOME=/usr/local/cuda-12.4 CDB_EXACT_BUILD_DIR="$CDB_REBUILD_OUT" PYTHONPATH=$PWD /home/yypan/CDB-AV/.venv/bin/python experiments/cdb_exact_qknorm/build.py
```

CUDA12.4/GCC11为已有工具；没有Ninja，使用显式nvcc/g++命令。Torch2.9.1+cu129/SM89，
源码和.so receipt必须匹配；其他工具链/硬件未验收。新构建完成后先独立/模型校验，再把
CDB_AV_EXACT_QK_LIB显式指向新.so；不要偷偷覆盖build或build_v2。
