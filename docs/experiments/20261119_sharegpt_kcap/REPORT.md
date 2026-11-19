# Qwen CDB HF 最大 K4/K8 配对实验（2026-11-19）

H8、HF0.8和既有底层开关固定，仅将服务端最大 speculative tokens 从8改为4。cap4仍是请求级HF动态K1–4，不是所有请求固定K4。

## 结果

6轮均512/512成功、每轮101,003正式输出tokens，共3,072请求、606,018输出tokens。标准差为样本SD；延迟列是每轮P95的算术均值，不是合并样本P95。

| 最大K | seed42 TPS | seed43 TPS | seed44 TPS | TPS均值±SD | E2E P95均值(s) | TTFT P95均值(s) |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 231.771 | 230.706 | 229.065 | 230.514 ± 1.364 | 26.160 | 0.246 |
| 8 | 229.056 | 230.678 | 228.297 | 229.343 ± 1.216 | 27.182 | 0.661 |

三组配对吞吐变化：+1.186% / +0.012% / +0.336%；平均 +0.511% ± 0.606%（n=3）。cap4相对cap8，均值之比约+0.51%，E2E P95均值约-3.76%，TTFT P95约-62.75%。吞吐收益小，尚不能声称稳定消除与原生K1的差距或对所有负载更优。cap4对本负载的延迟有更一致的改善。

## 配置与追溯

- 运行源码：`673c0c422e743aa39689ce6c65ec391047e28292`，worktree `/home/yypan/CDB-AV/worktrees/cdb-cpu-hotpath-20261113`，未修改生产代码。
- Qwen3-8B Target / Qwen3-1.7B Draft；GPU3 RTX4090；BF16/FA2；TP1；max_num_seqs4 / max_model_len1024 / max_num_batched_tokens1024；async与prefix caching关闭；outer enforce-eager配合手动Draft/Target Graph。
- 现有Pinned inputs、批量fingerprint、同调用父级KV ownership检查复用、精确QK both开启；post Graph关闭。H8、HF0.8、Graph cache24/minhits8、Draft ragged query bound8不变。
- 冻结ShareGPT 512条，输出预算16–384，客户端闭环并发16。warmup每臂128请求/C16/128输出。正式Source顺序与原生同seed plan SHA完全一致；客户端顶层vllm_xargs只传trace key，未设置请求K标签。
- 运行顺序：seed42 cap8→4，43 cap4→8，44 cap8→4。每臂新服务，正式测量重型校验关闭；同一低开销观察器和NVML采样。
- 改服务器cap也改变Target Graph最大query域9→5。因此本轮是完整配置对照，不能把全部差异归因于候选截短。实际启动命令、环境、Python导入路径、数据/源码/脚本SHA逐臂保存于run目录；交付附6个launch脚本。

## 工作量（三轮正式差分总和）

| 指标 | cap4 | cap8 |
|---|---:|---:|
| Target调用 | 42362 | 44615 |
| Draft逻辑Forward | 51267 | 49130 |
| 候选验证请求事件 | 74962 | 55641 |
| 候选tokens | 195641 | 189966 |
| Target接受Draft tokens | 139766 | 128810 |
| 平均验证K | 2.6099 | 3.4141 |
| 候选接受率 | 0.7144 | 0.6781 |
| 每Target调用验证请求数 | 1.7696 | 1.2471 |
| Draft平均有效请求行 | 3.8161 | 3.8666 |

cap4接受率提高、Target调用减少，但Draft Forward仍增加约4.35%；较短K需要更多轮次，收益会被额外工作抵消。有效Draft行均约3.8/4，不能把剩余差距都归因于空槽。计数不是GPU耗时，仍需真实trace定位临界路径。

## 与原生K1的差距预算

首组同seed42：原生K1 272.565 tok/s、Target14,370/Draft14,370；CDB cap4 231.771 tok/s、Target14,100/Draft17,138。CDB每次Target对应7.163最终输出token，原生7.029，CDB已略减少Target调用；但总耗时435.787 vs370.565秒。需要约15%壁钟缩减追平这条参考。

`总壁钟/Target调用`为30.907 vs25.787ms，含Draft/CPU/等待，绝不是Target GPU Forward时间。保持CDB当前输出/Target调用，需要将这个等效周期降至约26.28ms。此为差距预算，不是已定位出4.63ms CPU开销。原生Qwen位于GPU1，CDB位于GPU3，编译路径也不同；物理卡、时钟、CPU负载与完成顺序不能完全控制。原生不重跑。

原生三轮计数还显示Qwen K1/2/4接受率77.59%/68.97%/55.15%；对应Llama81.00%/73.23%/60.46%。差异存在，但K4每验证平均接受2.206 vs2.418，远不足以单独解释155 vs341 tok/s。Draft模型1.7B与1B、模型结构及执行路径也不同，不能将跨模型速度差全部归因于接受率。

## HF错配只读审计

16个短功能请求、cap8共232次验证；按真实request ID、context fingerprint和候选token完全一致关联Ready与Target结果，56次生成越过首次拒绝位置，共150个后续候选。它们是请求token行，不等于150次可省GPU Forward。70次低置信度停止后候选全部接受，仅是潜在早停信号，未生成后缀通常未知。

与独立固定K8短参考按trace key、完整上下文fingerprint/长度、Prompt/输出预算严格对齐：cap4共43个同上下文、40个同候选前缀且已有跨度接受一致；其中6个低置信度早停可在参考看到更长接受，按当前cap4限制合计11个额外输出机会。cap8对应34/30个对齐上下文，5个早停案例、12个额外机会；终止预算无收益的案例排除。

有实际cap8案例：8候选、末步置信度0.99962、Target首个即拒绝；也有cap4在K2、置信度0.7198停止，但独立同上下文K8接受8的案例（当前cap4内只算额外2个，不算6个）。这些证据说明Draft自信不等于Target接受；不能从小样本推出512请求总体误判率，独立batch/query形状也不构成严格反事实。平均K实验首组反而慢4.4%，不能仅凭存在误判就认定简单替换能提速。

## 功能与限制

216项CPU回归、3项协议/观察器测试通过。两个独立16请求GPU门槛共32/32，515交接=515验证，实际K分别1–4与1–8，均有low_confidence和max_k停止；87次跨Target Carry KV字节检查通过；结束Lease/Ready/inflight均0。

正式cap4/cap8整段输出逐Token相同的请求为174/169/159（每组512）；差异未分类，不宣称严格输出等价或全局最优。当前整体仍为Partial Implementation。

本轮6个服务均已关闭，每臂端口18084关闭、无本轮GPU进程、释放后15MiB；随后GPU3由独立Llama迁移实验使用。未停止GPU1其他服务，没有删除历史结果、备份、日志或改动他人分支。

原始结果：`/home/yypan/CDB-AV/results/sharegpt_kcap_gpu3_20261119`；代码目录软链接`cdb_av_sharegpt_kcap_results`。运行原始命令前务必确认GPU/端口空闲；不要在历史结果目录覆盖重跑。回退cap只调整speculative-config中的num_speculative_tokens到8，HF0.8与H8不变。
