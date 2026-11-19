# CDB-AV ShareGPT H=4/6/8 隔离比较（2026-11-19）

本轮没有证据支持以缩短 H 提高吞吐。保留 H=8；H=4 降低部分完成延迟，但 TTFT 明显增加。这里的 H 是固定 Draft 窗口，不改变请求级 HF 置信度策略。

## 源码与负载

- Worktree：`/home/yypan/CDB-AV/worktrees/cdb-cpu-hotpath-20261113`；提交 `673c0c422e743aa39689ce6c65ec391047e28292`。无生产代码修改。
- GPU3 RTX4090；Qwen3-8B Target、Qwen3-1.7B Draft；BF16 / FA2；Target/Draft 手动 Graph，外层 enforce-eager；异步调度关闭；max_num_seqs=4、max_model_len=1024、max_num_batched_tokens=1024、K cap=8、HF confidence=0.8。
- 冻结 ShareGPT 512 请求，闭环客户端并发16，输出预算16–384，101,003 output tokens/轮；不是此前统一128输出负载。客户端并发不等于 GPU Batch Size。
- 三种 H 各3轮，seed42顺序4→6→8，43为6→8→4，44为8→4→6。每轮独立服务，128请求/C16/128输出 warmup；seed42各臂还有相同8条兼容请求，不计入正式TPS。负载和协议 SHA 见 PLAN.json。
- 各轮完整启动命令在远端 `<run>/launch.sh`，本目录 launch.sh 是 H8/seed42 的实际命令副本。

## 实测结果

所有9轮均512/512完成、失败0，共4,608请求、909,027正式输出 tokens。标准差为样本标准差；延迟列是三轮各自P95的平均，不是合并样本P95。

| H | Seed42 TPS | Seed43 TPS | Seed44 TPS | TPS均值±SD | E2E P95均值(s) | TTFT P95均值(s) |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 224.958 | 227.329 | 226.685 | 226.324 ± 1.226 | 25.890 | 4.293 |
| 6 | 226.951 | 227.318 | 226.402 | 226.891 ± 0.461 | 27.981 | 1.766 |
| 8 | 227.238 | 229.396 | 228.247 | 228.294 ± 1.080 | 27.659 | 0.671 |

## 工作量与解释

三轮 Prometheus 测量前后差分：

| H | 参与验证请求事件 | Draft候选 | 接受Draft tokens | Target iterations | 平均验证K | 每Target iteration验证请求数 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 51810 | 177784 | 120845 | 46242 | 3.431 | 1.120 |
| 6 | 52878 | 180688 | 122977 | 45799 | 3.417 | 1.155 |
| 8 | 55361 | 189674 | 129175 | 44562 | 3.426 | 1.242 |

H4/H6/H8 平均验证K约3.43/3.42/3.43，接受率约68%，缩短H没有明显缩短K，因为Carry可以跨窗口完成。H4比H8的Target次数更多，单次Target容纳验证请求数更少，正式接受Draft token更少。这是吞吐未改善的工作量证据，尚不能仅凭计数分解全部壁钟开销。

正式区间内部聚合日志（不含边界尾部）：H4/H6/H8 Target Graph replay比例约76.1%/72.6%/70.1%，捕获次数均0。Graph覆盖率更高仍没有更高TPS，不能把命中率当作端到端收益。旧observer的Draft调用hook不能完整计入手动Graph replay，本报告不以其Draft次数推算计算量。

H4相对H8每个seed TPS分别下降1.003%/0.901%/0.684%；H6下降0.126%/0.906%/0.808%。只有3轮，不作广泛统计或全局最优结论。

## 功能、安全和边界

- 各H独立8请求GPU功能门槛，总24/24；H4/H6/H8共40/19/6次Carry KV逐字节检查，其中35/19/6次确实跨Target Forward；总65检查，60次真实Target interleave。
- 各门槛结束Lease/Ready/inflight及released_blocks计数均0，候选交接与验证一一匹配，Worker实际K涵盖1–8。
- 门槛输出H4 vs H8有7/8逐Token一致，H6 vs H8为8/8；未分类差异保留于FUNCTIONAL_OUTPUT_COMPARISON.json，不声称严格输出等价。HTTP成功、KV字节不变和输出严格等价是不同验收项。
- 所有本轮服务已结束并核对端口18084关闭、无本轮GPU进程、显存回到15MiB（之后由独立历史平均K项目使用）；没有停止GPU1原生服务或重跑原生基线。
- 原始证据：`/home/yypan/CDB-AV/results/sharegpt_h_sweep_gpu3_20261119_1016`；代码目录已有软连接 `cdb_av_sharegpt_h_sweep_results`。本目录JSON为摘要副本，原始日志不删。
- 不将本轮结果与原生GPU1实验解释成同物理卡、同编译栈因果对照；也不与统一128输出负载的约242 TPS直接混比。

## 后续

同一H8、同一底层版本下，用请求历史接受均值替换HF作独立对照。参考采集不计入TPS；每个请求必须有真实验证观测，缺失不当成0或假装命中。该实验是离线事后信息消融，不是可在线部署策略或逐轮最优上界。
