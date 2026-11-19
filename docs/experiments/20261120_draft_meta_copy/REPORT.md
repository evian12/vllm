# Draft Graph Metadata 整数拷贝融合：实现和 GPU 功能验收

代码 `a22f968f684d79f5d79b974236bb63d53a20953d`，分支 `codex/cdb-draft-meta-copy-20261120`；CPU、真实 CUDA 原语和模型
功能门槛通过；六轮512请求配对性能实验正在运行，**尚未证明吞吐提升，不作为推荐开启项**。
上轮 Target Spec Metadata 合并传输三组结果无吞吐收益，已保持关闭；本轮只研究另一条
Draft Graph 回放前的 Metadata 拷贝链路，不混入上轮开关。

新增 `cdb_av_draft_meta_copy.py`：保持整数 dtype，将 query offsets、seq_lens、
block table 清零与拷贝、slot mapping 合并为一个真实 Triton GPU 内核。
`cdb_av_refill.py` 只在独立开关 `CDB_AV_DRAFT_META_COPY=1` 下调用；默认0保留
原有 zero_/copy_。CPU query shadow 的更新仍在原路径，模型输入、位置、Forward
请求集合、KV Cache 更新及所有权管理、调度、HF、H 和提交逻辑均未改动。所有原安全断言保留。
整个源/目标 shape、dtype、device、contiguity 和范围在 GPU 写入前检查；不吞异常。
copy_ 调用数与实际物理传输次数不混为一谈，尚无 profiler 对新路径的性能归因。

CPU：**308 passed、8 skipped**；新增26项全部通过，版本工具15项通过。
8项是原有 pinned-input GPU opt-in 测试，本次未设置其独立启用变量，不能计为通过。
回归覆盖 Graph/config、pinned inputs、KV reuse、KV ownership 和 persistent queue。

真实 CUDA 原语 **160例**：1–4请求，混合 query 长度，1–1024列，完整及补零表、
int64大槽值/负值；四个字段与独立原生 zero_/copy_ 结果逐字节一致，输入未改变，
160份输出延迟核对通过。这是整数 Metadata 测试，未伪造模型/KV 上下文作为系统证据。

实际 Llama 功能测试64预热+16短请求全部完成；**706 Draft Graph**、
**192 Target Graph** 检查通过；**216跨 Target Carry KV 字节检查**通过，
**1202次唯一候选交接=1202次验证**；最终 lease/Ready/inflight/released_blocks 均0。
32个实际模型拷贝审计中四个字段全部精确一致，Worker 日志记录的实际模块路径和
源码哈希对应此分支。K实际覆盖1–8，停止原因包括low_confidence/max_k。
独立 CUDA 进程和功能服务各自退出后GPU3均回到15MiB、18084关闭、无本轮遗留PID。

正式比较：Llama8B/1B，ShareGPT512/C16、101003输出token/轮，max_num_seqs4、
len1024/budget1024，BF16/FA2/TP1/no-async/no-prefix-cache/eager outer+manual Graph，
H8、HF0.8、cap8；Draft sequence floor1024固定，Target bound reuse/GEMM/Target Spec
Metadata packing均关闭，仅本轮拷贝融合0/1不同。三组seed42/43/44交替顺序，
128请求预热排除；重型拷贝/Graph/KV审计在性能阶段关闭。每轮 driver 在 finally
仅回收本次服务，并核查端口、GPU PID和显存。失败即停止，不结束其他实验。

完整启动命令见 gate_fused/launch.sh 和各正式轮次 launch.sh，包含所有环境变量、
GPU3、模型、端口18084及Serve参数；当前服务仍在测量，不重复启动。
回滚使用同一代码的 `CDB_AV_DRAFT_META_COPY=0`，保留其它固定配置，不执行git reset。
正式结果以服务器同名结果目录的 COMPLETE.json、ANALYSIS.json、released.json 为准，
此文件是功能验收时的快照。目标341.073279±5.496631 tok/s仍未达到；完整输出严格
等价性不由Metadata门槛单独证明，最终配对必须继续保留输出差异记录。

补充只读审查：TP1/world_size1且无显式分布式后端时，parallel.py默认选择uni；
uniproc_executor.py的collective_rpc直接run_method，不应把该函数调用次数当成网络或
进程间RPC次数。当前启动命令没有覆盖分布式后端；这是配置解析/源码证据，尚未另加
运行时executor类名探针。API到EngineCore的进程边界仍存在，不与Worker调用混淆。

原始日志、请求和结果留在 `/home/yypan/CDB-AV/results/draft_meta_copy_20261120`；
源码目录和主vllm目录均有 `cdb_av_draft_meta_copy_results` 软链接；不清理历史证据。
