# CDB-AV：尾部输出预算上限实现与性能验证

运行码ff6a36e3680092b0046164c26e6bcc94709ecfc0；分支codex/cdb-output-budget-20261106。代码、CPU测试与GPU功能通过；吞吐收益未得到充分确认，整体仍Partial。

## 变更与实际执行

新增默认关闭的CDB_AV_OUTPUT_BUDGET_CAP。cdb_av_config.py计算min(policy_cap,max(0,max_tokens-output_tokens-1))；cdb_av_persistent.py在策略cap之后应用物理预算；Persistent/Reuse两处plans过滤0。0只表示CPU端AR-only尾部，不发送Worker。cdb_av_kv_reuse.py仅增加这两行过滤；fill/fill_batch/reuse_span、证书、释放、失效、Scheduler均未改。HF0.8、H8保持；Carry正式前缀冻结，原cap跨窗口保留。

长边界请求Prompt301/max_output16，末轮context313只余4输出：实际Worker由K8改K3、Target由9个输出改4个。新请求及已有缓存请求只余1时均观察到真实native AR query1，无Worker cap0。四个OFF/ON受控请求返回的Token IDs全部完全一致。CPU cap决策与Worker Ready/progress、实际Target行分开审计，未把标签当GPU证据。

## 功能与回归

- 301功能测试+12工作流通过；checkpoint再次通过74相关功能测试+12工作流。新增26项覆盖预算/类型/默认OFF/两Queue的0cap过滤，未放宽原断言。
- GPU32/32：223次Target对照，36层当前写入K/V逐字节一致、Token相同、原allclose通过；1107次验证覆盖拒绝315/部分接受261/全接受531，32个请求全部能关联正式预算，生成结果数均不超剩余预算，Worker cap均正。
- Carry16/16：38次真实跨Target恢复，28层Draft KV累计343146496字节一致；取消4个流后恢复4/4，138次Target对照。Lease/Ready/inflight最终0、reset额外释放0。
- 另有四个受控请求的26次Target对照；总计387次，不代表所有布局/自然内存压力/严格512输出等价均已认证。
- 原生Scheduler CPU子集未通过：17失败/1skip/63未选。起初缺OPT/LLaVA配置；通过官方内容和Git blob哈希核验，仅在隔离测试缓存恢复原始config.json后，仍缺其Tokenizer。未换Tokenizer、未改原测试预期。使用项目本地Qwen3配置另做native finish/preempt+延迟结果处理CPU回归，2/2通过；不冒充原子集通过。

原始配置来源：[OPT固定版本](https://huggingface.co/facebook/opt-125m/resolve/27dcfa74d334bc871f3234de431e71c6eeba5dd6/config.json)、[LLaVA固定版本](https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/config.json)。原配置Git blob和SHA256见hf_test_configs/VERIFIED_SOURCES.json，未下载权重或改变运行模型。

## 公平性能对照

同一ff6a36e3/GPU3/18084/Qwen3-8B+1.7B/BF16/FLASH_ATTN FA2。两组均Target GraphON、dynamic queryON、sequence floor0、Draft query4/cache12/minhits8、Target cache24/minhits8、H8/HF0.8/maxK8、max_num_seqs4/max_model_len1024、seed42。4个相同warmup后，512请求/C16闭环/输出128。详细trace/字节验证均OFF，两组NVML采样一致。客户端16不是GPU Batch16；真实到达顺序/批次组成没有强制相同，未锁GPU时钟。

|轮次/顺序|OFF tok/s|ON tok/s|变化|OFF P95 s|ON P95 s|
|---|---:|---:|---:|---:|---:|
|1/OFF→ON|218.865|222.797|+1.797%|11.233|11.207|
|2/ON→OFF|223.364|223.213|-0.068%|11.286|11.296|
|3/OFF→ON|221.416|221.981|+0.255%|11.172|11.147|

**OFF221.215±2.256，ON222.664±0.627 tok/s（样本标准差）；均值+0.655%。各轮P95均值11.230→11.217秒。两组各3×512请求全部成功，每轮65,536输出Token。首组+1.80%未在全部轮次重复，不据此宣称稳定加速，开关保持默认OFF。**

32请求三组：197.961±4.447→201.288±2.052（+1.68%），其中OFF第三轮192.857偏低，原样保留；不用该短测均值证明收益。没有重跑AR/原生SD，也不把其他版本/GPU/时间的绝对TPS混入配对。
NVML模式0各轮样本均值的均值：util 76.342%，graphics 2670.452 MHz，power 239.163 W。
NVML模式1各轮样本均值的均值：util 75.847%，graphics 2670.136 MHz，power 239.688 W。

NVML首区间剔除，N/A/非有限值不当0；util不等于Kernel时间，100-util不等于CPU开销。原生TTFT/TPOT包含每轮4个预热；ITL按输出事件，一次可有多个Token。周期性Target/CDB计数仅为最后记录的累计前缀，不当全程Forward计时。文本hash配对结果在SUMMARY512.json，不能据此认证或归类所有数值差异。

## 根因与下一项测量

确认了尾部无用计算，但节省Token行没有带来稳定等比例加速。可能受小Batch权重访问、Target/Draft阶段工作量、AR尾部批处理、CPU同步/元数据等影响；目前没有分阶段计时，未选定唯一根因。上一项sequence floor256虽提高replay覆盖，512仅+0.335%且P95未改善，保持floor0。

当前TP1默认选择UniProcExecutor：parallel.py设置uni，executor/abstract.py选择该类，其collective_rpc直接run_method(driver_worker)。Worker接口独立，但当前不是每步跨进程网络往返；不能把RPC总elapsed都归为通信。下一步在独立诊断分支对同一冻结Forward测Graph/static-eager/live-eager的CUDA stream span、CPU提交与6个Buffer更新，同时保留字节/Token断言。GPU事件跨度包括Host提交间隙，不伪称CUPTI纯Kernel时间。

## 复现与回滚

完整命令/环境见LAUNCH.md、start_budget0.sh、start_budget1.sh和pair512/*configuration.json。pair32.py/pair512.py自动预检、等量预热、balanced顺序、finally仅关闭自己服务；拒绝覆盖旧OUT，复测请新建结果目录。解析：python3 analyze_performance.py pair512 --output NEW_SUMMARY.json；python3 analyze_nvml.py pair512 --output NEW_NVML.json。

回滚CDB_AV_OUTPUT_BUDGET_CAP=0/unset；不执行reset/覆盖checkout/clean/force push。原树、备份、历史结果保留。源码及结果已软链到代码目录。GPU结束后显存15MiB、18084关闭，见CLEANUP.json；后续诊断若启动会单独记录并关闭。
