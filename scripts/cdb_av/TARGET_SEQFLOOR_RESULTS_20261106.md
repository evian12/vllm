# Target 序列长度桶合并：GPU功能通过，吞吐收益未确认

运行码：019f3d150782025d751d35ac908a18ba3300322f；基于2169190。独立分支codex/cdb-target-seqfloor-20261106。

仅增加默认关闭的CDB_AV_TARGET_GRAPH_SEQ_FLOOR。floor0保留原序列桶；floor256将短上下文的捕获长度上界合并到256。实际Token/请求行、seq_lens、位置、槽位与Block Table不变，未修改H、HF算法、KV reuse/租约或调度。

两组均Target backbone Graph ON、query动态上界ON，Draft query4/cache12/minhits8，Target cache24/minhits8；H8、HF0.8、maxK8、max_num_seqs4、max_model_len1024、BF16/FA2、GPU3/18084。每轮4个相同HTTP预热；C16闭环、512请求、每请求128输出。重型验证/详细trace均关闭，两边以相同NVML采样器只观测客户端过程。

|轮次/顺序|floor0 tok/s|floor256 tok/s|变化|floor0 P95 s|floor256 P95 s|
|---|---:|---:|---:|---:|---:|
|1/0→256|222.850|222.877|+0.012%|11.095|11.129|
|2/256→0|221.328|221.749|+0.190%|11.249|11.401|
|3/0→256|223.004|224.793|+0.802%|11.232|11.084|

**最终：222.394±0.926 → 223.140±1.539 tok/s（样本标准差），均值+0.335%。各轮P95均值11.192→11.205秒。两组各3×512全部成功、每轮65,536输出Token。幅度小于波动，不认定获得吞吐提升，不推广floor256，默认0保留。**

32请求三组短测199.846±0.572→202.397±2.399，+1.276%；未用短测代替长测。只与同提交floor0配对比较，未重跑AR/原生SD，不将上轮绝对TPS混入本轮。真实HTTP到达顺序/批次组成仍受线程和运行速度影响，未锁GPU时钟。

首组最后9,216次Target调用前缀：replay 6,700→8,357（72.7%→90.7%），容量回退1,790→195。命中改善是真实的，但不是GPU时间占比，也没有转化为可确认的端到端收益。更宽FA2长度上界的Kernel成本、Metadata/LM head、RPC等可能抵消；尚无分阶段计时，不能选定根因。
floor0 NVML各轮样本均值的均值：GPU util 76.494%，graphics 2665.497 MHz，power 240.458 W。
floor256 NVML各轮样本均值的均值：GPU util 76.465%，graphics 2667.085 MHz，power 241.335 W。

NVML丢弃首个可能跨预热的平均区间，N/A/非有限值按不可用处理；这些是1秒观察样本，不是Kernel时间，也不能用100-util作为CPU开销。原生TTFT/TPOT直方图包含每轮4个warmup；原生ITL按输出事件记录，一次可输出多个推测Token，不能当作客户端逐Token ITL。

## 正确性门槛

- CPU261功能测试（Target63、config130、Draft Graph34、queue23、ownership11）及12工作流通过。新增13个域/不变性测试；初始query1夹具误用mixed9被旧断言拒绝，修正夹具依照真实key的query域，没有放宽断言。
- smoke4/4：20次Target对照；32/32：519次对照、137次同键实际长度桶变化；实际max_query=1..9。全部36层当前写入K/V字节一致，greedy Token一致，logits通过原有比较；对照均包含静态与原生live eager。
- 短/长边界2/2：Prompt实际1和301 Token，10次对照；捕获上界256/512，位置包含0及302..320。超出最终输出预算的原生lookahead位置不代表额外正式提交。
- Carry16/16：44次跨真实Target阶段恢复，28层Draft KV累计407371776字节一致；取消4个流后4个恢复完成，138次Target对照。最终Lease/Ready/inflight均0、reset未释放额外残留块。
- 687次Target对照只覆盖发生的布局/位置，不能认证所有36行组合、自然内存压力或严格全生成等价。系统仍Partial。

## 发现的下一项确定性浪费

边界长请求prompt301/max_output16，末轮context313意味着只余4输出；Worker仍cap8/actual8，Target返回9个候选验证结果，最终API只返16总Token，末轮5个结果不用。CDB resolve_k_cap目前没有输出剩余预算上限。应单独接入物理预算clamp（保留HF0.8）；只余1 Token时走native AR，不能把0 cap发给Worker。Carry已冻结正式前缀，预算clamp只作用于新admission；不改lease/commit/reuse规则。当前floor对照没有包含此修改。

## 复现/回滚/资源

完整启动环境与serve命令：LAUNCH.md、start_floor0.sh、start_floor256.sh；逐轮实际JSON见pair512/*configuration.json。pair32.py/pair512.py保存预检、等量warmup、balanced顺序与finally关闭自己服务；既有输出目录会被拒绝覆盖，复测须另取新OUT。解析：python3 analyze_performance.py pair512 --output NEW_SUMMARY.json；NVML：python3 analyze_nvml.py pair512 --output NEW_NVML.json。

回滚：CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0或unset，不执行覆盖式Git命令。服务均仅测试期间启动，最后GPU3回15MiB、18084关闭，见CLEANUP.json。其他服务/历史数据/未提交文件/基线保留。代码019f已正常push并核对；功能通过与性能未确认分开报告。

解析器修复记录：客户端分别把wall_time和TPS格式化到3位小数，固定绝对误差0.003会误拒短测的合法舍入结果。现按两项的±0.0005量化区间验证是否相交；未改原始客户端日志/模型比较/性能数据。旧解析器保留。
