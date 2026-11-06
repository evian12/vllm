> Raw evidence and full startup scripts are on the server under `/home/yypan/CDB-AV/results/target_dynamic_20261106/` and through the code-directory result symlink. Raw logs are intentionally outside Git.

# CDB-AV：Target backbone Graph 与混合 Query 域

日期：2026-11-06。当前交付为实验分支，整体系统仍为 **Partial Implementation**。

同提交512请求配对各3次：214.438 → 221.427 output tok/s，均值 +3.26%；全部3,072个测量请求完成。所有运行保持 H=8、HF confidence=0.8，不修改请求选择、补位、KV reuse 或租约逻辑。GPU实验结束后本次服务已关闭，GPU3回到15 MiB。

## 实现与版本

- `2523fdd75d75fbd9223c8eb46172a58b1f9ed511`：增加默认关闭的手工 Target backbone CUDA Graph。保持 `--enforce-eager`，不引入全局 torch.compile 或 fusion 改变。
- `14b225db376bb7c5e80cf9924472bee69ef024a5`：增加默认关闭的混合 Query 上界域。实际 query=1 独立；query=2..9 共用静态上界9，但实际 Token/请求行数不增加。
- 源码目录：`/home/yypan/CDB-AV/worktrees/cdb-target-dynamic-20261106`。
- 原始结果：`/home/yypan/CDB-AV/results/target_dynamic_20261106`。

修改的运行时文件为 `vllm/v1/spec_decode/cdb_av_target_graph.py` 与 `vllm/v1/worker/gpu_model_runner.py`（9行入口）。测试位于 `tests/v1/spec_decode/test_cdb_av_target_graph.py`，配置与范围写入 `scripts/cdb_av/TARGET_FULL_GRAPH.md` 和 `TARGET_DYNAMIC_GRAPH.md`。

Graph 覆盖 embedding、全部36层 Qwen3-8B transformer 和 final norm。LM head、logits 行选择、采样、拒绝处理、正式 Commit、Scheduler 仍走原有路径。Draft 模型为28层，不能混用层数。

捕获键保留实际 Token 行数、请求行数、Query 域、序列长度桶、Block Table 宽度和 split 参数。每次 replay 复制当前 Input IDs、Positions、Query Starts、Sequence Lengths、Block Table 和 KV Slots；无伪造 uniform 标志，无额外 Padding 行。模型/Attention 身份变化、KV 存储变化都有显式防护；不支持的布局在 Forward 前回退。保留现有 KV 碰撞、上下文与进度断言。

## GPU 正确性证据

混合域版本：

- 4/4 smoke，18次 Target Graph 对照。
- 32/32 功能测试，162次对照，覆盖实际最大 query=1..9、99次键内布局变化、55次键内最大 Query 变化、137次键内请求成员变化。
- 每次对照比较静态 Metadata eager 与原生 live Metadata eager：全部36层当前写入位置的 K/V 逐字节一致，greedy Token 相同，logits 满足既有 allclose 检查。额外两次 eager Forward 只在功能审计启用；性能测试关闭。
- 16/16 Carry 测试，48次跨真实 Target 阶段恢复、28层 Draft KV、累计442,236,928字节一致；最终 Lease/Ready/inflight=0。
- 4个真实流断连后，4个恢复请求完成；138次 Target 对照通过，Lease 逐步3→2→1→0，最终0。
- CPU 功能测试248项、版本工作流测试12项通过。

GPU trace 没有观察到同一键内“实际最大 Query≤4 直接切到9”的特定边界；该组合只有 CPU 协议测试。不能把已见的1..9覆盖推成所有布局组合均已验收。

## 公平性能口径

RTX4090 物理 GPU3、端口18084；Qwen3-8B / Qwen3-1.7B，BF16、FLASH_ATTN FA2、TP/PP/DP/DCP=1、max_num_seqs=4、max_model_len=1024、服务端 maxK=8、H=8、HF0.8。Draft query4 Graph、cache12/minhits8；Target cache24/minhits8。四个相同短请求预热后，使用相同固定并发客户端、同序号 Prompt、greedy/seed42、输出128。每个服务单独启动并在该轮结束后关闭。

ON/OFF 使用相同14b225d代码，只切换 Target Graph；详细 Trace、Metadata/KV 校验与重复 eager 对照均关闭。客户端并发16是闭环在途请求数，不是 GPU Batch=16。实际 Target 最大请求行仍为4。

短测32请求、每组3次：OFF `200.934 ± 1.196`，ON `202.554 ± 1.932` output tok/s（样本标准差），均值变化 +0.806%。幅度小，需要更长配对验证。

前一版精确 Query 最大值键三次短测为 OFF `199.100 ± 2.861`、ON `198.064 ± 2.941`，变化−0.520%，没有作为性能改进推广。更早 native PIECEWISE 路线均值+0.08%，同样没有证明稳定收益。这些是不同实验，不与新版本绝对 TPS 混成配对。


## 512请求最终结果

| 轮次／顺序 | Target Graph OFF tok/s | ON tok/s | 配对变化 | OFF P95 s | ON P95 s |
|---|---:|---:|---:|---:|---:|
| 1／OFF→ON | 211.000 | 221.912 | +5.17% | 12.305 | 11.414 |
| 2／ON→OFF | 216.094 | 223.619 | +3.48% | 11.461 | 11.123 |
| 3／OFF→ON | 216.221 | 218.749 | +1.17% | 11.507 | 11.448 |

两组均为3×512请求、每轮65,536个输出Token，失败0。OFF均值 **214.438 ± 2.978**，ON **221.427 ± 2.471** tok/s（样本标准差）；均值比变化 **+3.259%**。各轮P95均值11.758→11.328秒。

合并各组1,536条客户端请求记录：平均延迟 9.505→9.209秒，P50 9.613→9.299秒，P95 11.701→11.389秒。客户端延迟从工作线程实际发起请求计时，不含线程池等待。

三组均正向，但变化5.17%、3.48%、1.17%，不能用首轮5.17%代替最终结果，也不宣称普适或统计显著。共享服务器未锁GPU时钟、未固定真实HTTP到达顺序、未连续采集GPU时序；并发线程的实际到达和后续Batch组成会变化。没有重新运行AR或原生SD，本轮只证明这项CDB实现增量在当前负载下的实测变化。

原生Prometheus指标包含每轮4个预热请求（516请求／轮），只作为辅助：

| 原生服务指标均值 | OFF | ON |
|---|---:|---:|
| TTFT ms | 119.378 | 116.866 |
| 请求TPOT ms | 73.446 | 71.150 |
| 输出事件间隔 ms | 135.770 | 131.125 |

当前vLLM `stats.py` 在每个后续输出事件记录一次 `engine_core_timestamp-last_token_ts`，推测解码一次可输出多个Token，因此该ITL直方图不是客户端逐Token流式ITL。客户端为非流式，不能据此报告真实逐Token到达分布。没有去除预热Prompt Token的可靠计数，未推算Total Tokens/s。

ON最后一条周期性Target计数均覆盖前9,216次调用：replay为6,646／6,697／6,813，capture均24，容量回退1,812／1,775／1,647。既有缓存命中仍留下约18–20%的容量回退；这不是回退的GPU时间占比。短测捕获与预热成本更难摊薄，长测收益更大符合该机制，但尚无分阶段时间证明全部归因。

第一组最后CDB累计前缀中，实际K=1..8、均值约2.43，低置信度停止为主；已有KV Token重用比例约96.64%。该比例为 `reused/(reused+prefill)` 的Token工作量口径，不是内存命中率或时间收益。Worker累计证据与HTTP客户端requested_K严格分开。

三组OFF/ON文本哈希完全一致为159/512、156/512、154/512；已保留逐请求哈希，未把其余差异归类为无害浮点差异，也未据此宣称严格最终输出等价。

## 已识别的剩余瓶颈与证据边界

1. **捕获键碎片与容量回退。** 32请求功能 Trace 出现77个不同键，序列长度桶64/128/256会拆开相同实际行数。离线合并上界256的模拟可将该 Trace 降为36键；这只是缓存模拟，未改变 GPU Forward，也不是性能结果。应先验证 FA2 对更宽序列上界的执行成本与正确性，再判断能否合并。
2. **未捕获路径仍然存在。** 大 Query、复杂 Prefill 等布局仍回退；LM head 与采样在 Target Graph 外。需要分别测 GPU 时间和 Host 提交时间，不能因 Graph 命中率上升就断言它们是最大瓶颈。
3. **算法收益有上限。** HF0.8提前结束减少低置信度草稿，同时仍支付多次 Draft Forward 和 Target 阶段成本。动态 K、接受率与 KV 复用都应读取 Worker/Commit 证据；不能由客户端 `requested_K=8` 推断实际 K=8。
4. **现有累计计数不等于完整计时。** 最后一条周期性 Graph/CDB aggregate 仅覆盖该累计前缀，未覆盖日志尾部。重用 Token 比例不等于 GPU 时间节约比例；不能把原生 speculative proposal 计数当作 CDB GPU Forward 次数。

完整生成严格等价仍未认证。Graph 的逐次 KV/Token 对照通过，不能代替整个512请求执行轨迹的等价证明；文本哈希不一致也不能直接归类为浮点误差。自然内存压力、全部取消/抢占/异常组合仍需要扩大覆盖。

## 复现与回滚

交付目录保留 `start_eager_target.sh`、`start_bounded_target.sh` 的完整环境与 serve 命令，以及各轮配置、启动源码路径、请求响应摘要、原始日志、Manifest、哈希、审计与解析脚本。启动脚本不自动终止其他服务；运行前必须检查 GPU3 和18084。

- 回到同提交 eager Target：`CDB_AV_TARGET_FULL_GRAPH=0`。
- 保留精确 Query 最大值键：`CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_DYNAMIC=0`。
- 本次候选：`CDB_AV_TARGET_FULL_GRAPH=1 CDB_AV_TARGET_GRAPH_DYNAMIC=1`。

没有 reset、覆盖式 checkout、clean、force push、模型或 Tokenizer 修改。原始工作树、备份和历史实验保留。最终 Commit、远端 SHA 与 GPU 清理证据见下方最终结果。

## 清理与可见结果

2026-11-06 19:02:02 CST检查：18084和已停用的18082均未监听；GPU3显存15 MiB、无计算进程。其他GPU进程未被本轮终止。见 `evidence/CLEANUP.json`。

代码目录软链接已建立：

- `/home/yypan/CDB-AV/vllm/cdb_av_target_dynamic_results`
- `/home/yypan/CDB-AV/worktrees/cdb-target-dynamic-20261106/cdb_av_target_dynamic_results`

均指向本轮服务器结果目录。精确键版本也有对应 `cdb_av_target_full_results` 链接。
