# CDB-AV 固定 K8 参考与历史请求平均 K（2026-11-19）

本轮按用户后续决定停止平均 K 实验，保留 HF=0.8，转回底层优化。平均 K 没有在已有结果中带来提速；三组配对计划未完成，不能报告为三轮隔离结论。

## 实际完成结果

运行源码 `8c94d2ac950c2b4a46ff308a827cb0874a814d37`，基于底层优化提交673c0c；H8、K cap8、Qwen3-8B/1.7B、GPU3、FA2/BF16、max_num_seqs4、max_model_len1024、max_num_batched_tokens1024、async OFF、现有手动Draft/Target Graph。正式512请求/C16闭环、101,003输出tokens，128请求/C16/128输出预热。采集、功能验证和性能区间分开。

| 策略 | Seed | 成功/请求 | Output tok/s | E2E P95(s) | TTFT P95(s) |
|---|---|---|---:|---:|---:|
| HF 0.8 | 42 | 512/512 | 228.078 | 27.804 | 0.655 |
| 请求历史平均K | 42 | 512/512 | 218.033 | 28.895 | 1.197 |
| 请求历史平均K | 43 | 512/512 | 218.619 | 28.808 | 1.083 |

只有seed42构成完整配对：平均K吞吐-4.404%，E2E P95+3.924%。两个平均K轮次为218.326±0.414 tok/s（n=2，样本SD）；HF只有n=1，不把不同重复次数的均值差当作三组配对证据。两臂首组间隔约两小时，其他GPU/CPU负载及完成顺序未完全固定；GPU平均SM时钟几乎相同（约2669.5/2669.6MHz）。

## 参考轨迹的完整性

- 固定K8采集关闭HF提前停止。正式512/C16中437请求有9,220次候选验证；75请求没有接受记录。逐一关联CPU调度、请求ID及输出，75条确实全部只做了Prefill+AR，不是接受长度0，也不是日志丢失。它们输出预算16–274，合计5,783 tokens。
- 对这75条单独以原Prompt和原输出预算、固定K8、C1补采，75/75参与验证，增加1,631次真实验证、4,291接受Draft tokens。补采不计入吞吐对照。
- 合并后512/512有历史均值，共10,851次验证、34,210接受Draft tokens。只补缺，不改变原437条均值。取每请求 `ceil(accepted_total/verified_count)` 并clamp到1–8；真实零均值取最小合法K1，未知不冒充零。
- 选择K分布：1:7、2:60、3:180、4:149、5:63、6:30、7:9、8:14。每请求每轮固定用自己的历史均值，没有声称逐轮动态预测。
- 参考表SHA256：`27d862c583f5310e90048b0f157a2415d739972d74f1705b5bd6d3a189f52aab`。正式与暖机key分开，暖机只是同一正式历史值的别名，不增加历史观测。
- `accepted`是原生Target verifier的采样接受深度，不含bonus/correction；Scheduler随后仍可能做最终输出预算裁剪，末轮接受数不一定全部交付API。K8处还存在截断，不能知道是否本可接受超过8。

这是同请求事后数据消融，437条与75条参考采集batch shape不同。不是在线策略泛化、逐轮oracle或吞吐上界。同一请求接受长度的标准差中位数2.70；参考样本中56.6%的轮次接受数低于所选平均K，34.0%高于，平均不能追踪这些变化。

## 首组工作量证据

| 正式测量差分 | HF | 平均K |
|---|---:|---:|
| Target iterations | 14,915 | 16,007 |
| 参与候选验证的请求事件 | 18,563 | 17,415 |
| 候选tokens | 63,328 | 64,397 |
| Target接受Draft tokens | 42,844 | 37,871 |
| 接受率 | 67.65% | 58.81% |
| 平均验证K | 3.412 | 3.698 |
| Draft真实逻辑Forward次数（含Graph replay） | 16,423 | 16,679 |
| Draft平均有效请求行 | 3.856 | 3.861 |
| Draft总输入token行 | 152,328 | 158,268 |
| 纯AR Target调用 | 3,717 | 5,173 |
| Ready→Target schedule平均等待(ms) | 189.62 | 204.96 |

两个阶段计数边界交叉核验：Target调用计数等于Prometheus iteration count、Verify请求行等于num_drafts；不是把原生speculative metrics当作Draft全部计算量。Draft token行另由实际 `_forward` 的输入形状计数，包含必要上下文计算；捕获内部额外执行另属捕获成本。Ready等待到CPU schedule，不冒充GPU验证开始时间。

平均K的Target Graph内部区间replay约78.5%，HF约71.2%，捕获均0；覆盖率较高仍未抵消接受收益下降及更多Target调用。时钟几乎一致也不等于控制了所有环境因素。不能据此认定HF最优或差距全部来自某一底层组件。

## 实现、验证及限制

- 8c94d2ac只为既有历史K接口增加显式`CDB_AV_HISTORY_K_FALLBACK=hf_confidence|fixed_max`，默认保持HF回退；已匹配历史请求跳过confidence计算，增加Worker真实Ready来源聚合。没有修改权重、dtype、KV复用判定、租约或调度。
- HTTP参数走真实顶层`vllm_xargs.cdb_trace_key`；CPU建表以Prompt fingerprint核对；不以客户端标签充当GPU执行证据。完整参考表避免性能臂因未知请求使用默认K。实际Worker聚合来源为request_average；功能门槛另外覆盖固定回退，confidence=None验证跳过HF计算。
- 245项源码CPU回归通过。16条K8采集门槛+16条异长平均K门槛共32/32完成、184次真正跨Target Carry KV字节检查；候选交接与验证闭合，结束Lease/Ready/inflight均0。
- 补采1,631次交接与验证、75条最终租约释放逐一闭合；最终running/waiting/KV usage均0。补采未输出p2_reset，原脚本错误要求该日志而停；修复后保留reset=null，依靠明确事件账本、最终指标及进程退出证据恢复，未删除核心断言或虚构reset ACK。新增工具测试15项通过。
- 首组输出167/512整段逐Token一致，345条不同，分歧首位置中位数62、最早0。原因尚未分类；不一概归为浮点差异，不声明严格输出等价。当前系统仍是Partial Implementation。
- 新worktree首次启动因缺少未跟踪的FlashAttention运行库绑定失败；复用既有相同二进制和Python包装并记录SHA后启动成功，没有升级依赖或更换后端。

## 资源、复现和同步

原始目录 `/home/yypan/CDB-AV/results/request_avg_sharegpt_gpu3_20261119`，代码目录软链接 `cdb_av_request_avg_sharegpt_results`。三轮正式服务和两轮参考采集均关闭，端口18084关闭、无本轮GPU进程、每轮释放后15MiB。用户决定后续平均K不再开展，预检guard拦截下一服务，没有打断正在运行的一轮。原始退出日志保留，STATUS另注明stopped_by_user_selection。

完整实际启动命令见各run的launch.sh；交付目录提供三轮副本。原始请求/响应、warmup、环境、导入路径、哈希及日志均保留服务器。恢复HF只需选择`CDB_AV_K_POLICY=hf_confidence`；不必回滚历史结果。后续cap4/8实验另用底层源码673c0c，不将文档提交冒充运行源码。

源码8c94d2ac、H报告e163fde8、采集审计工具ab413313均已正常push并核对远端SHA。论文/生产推荐仍保留HF0.8；下一步是H固定的最大K4/8隔离对照和有界底层GPU trace，不把本轮未完成三组的结果包装成成功优化。
